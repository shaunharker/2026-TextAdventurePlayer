#!/usr/bin/env python3
"""
Web-based AI text adventure player.
Serves a browser UI for watching an AI play interactive fiction,
with live streaming, hint injection, and rollback support.
"""

import asyncio
import copy
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

import requests
from aiohttp import web

from main import (
    AgentConfig, FrotzEngine, ContextManager,
    TiktokenTokenizer, clean_game_output, extract_json_robust,
)

# ===========================================================================
# Configurable Prompts  (adjust freely — these are NOT the game LLM prompts)
# ===========================================================================

HINT_PREFIX = "HINT FROM OBSERVER"

TRANSLATOR_PROMPT = """\
You are a translation layer between a casual human observer and an AI \
playing a text adventure game. Rewrite the following casual hint as a \
formal in-game system suggestion. Be concise. Preserve all actionable \
content. Remove profanity and emotional language. Output only the \
translated message, prefixed with "SUGGESTION SYSTEM:".\
"""

GAMES_DIR = Path(__file__).parent / "games"


# ===========================================================================
# Exceptions
# ===========================================================================

class LLMCancelled(Exception):
    pass


# ===========================================================================
# Instrumented LLM Client
# ===========================================================================

class InstrumentedLLMClient:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.last_stats: Dict = {}
        self._streaming_response = None

    def chat_completion(self, messages, max_tokens, temperature) -> str:
        """Non-streaming call (used by compaction and hint translation)."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        start = time.monotonic()
        resp = requests.post(
            self.config.llm_url, json=payload, timeout=self.config.timeout
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        self.last_stats = {
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached,
            "uncached_tokens": prompt_tokens - cached,
            "output_tokens": output_tokens,
            "ttft_ms": round(elapsed_ms),
            "gen_time_ms": 0,
        }
        return data["choices"][0]["message"]["content"]

    def chat_completion_streaming(self, messages, max_tokens, temperature,
                                  cancel_event, chunk_callback) -> str:
        """Streaming call. Calls chunk_callback(text) per token. Cancellable."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start = time.monotonic()
        self._streaming_response = requests.post(
            self.config.llm_url, json=payload,
            timeout=self.config.timeout, stream=True,
        )
        self._streaming_response.raise_for_status()

        accumulated = ""
        usage = {}
        first_token_time = None

        try:
            for raw_line in self._streaming_response.iter_lines():
                if cancel_event.is_set():
                    raise LLMCancelled()
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        accumulated += content
                        chunk_callback(content)

                if "usage" in chunk:
                    usage = chunk["usage"]
        except LLMCancelled:
            raise
        except Exception:
            if cancel_event.is_set():
                raise LLMCancelled()
            raise
        finally:
            try:
                self._streaming_response.close()
            except Exception:
                pass
            self._streaming_response = None

        now = time.monotonic()
        ttft_ms = round((first_token_time - start) * 1000) if first_token_time else 0
        gen_time_ms = round((now - first_token_time) * 1000) if first_token_time else 0

        prompt_tokens = usage.get("prompt_tokens", 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        self.last_stats = {
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached,
            "uncached_tokens": prompt_tokens - cached,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "gen_time_ms": gen_time_ms,
        }
        return accumulated

    def cancel_streaming(self):
        resp = self._streaming_response
        if resp:
            try:
                resp.close()
            except Exception:
                pass


# ===========================================================================
# WebSocket Renderer
# ===========================================================================

class WebSocketRenderer:
    def __init__(self):
        self.clients: set = set()
        self._pending: List[dict] = []

    def session_start(self): pass
    def game_output(self, text: str): pass
    def turn_start(self, turn: int): pass
    def ai_action(self, reasoning: str, command: str): pass
    def session_end(self): pass

    def system_message(self, message: str):
        self._pending.append({"type": "system_message", "message": message})

    def error(self, message: str):
        self._pending.append({"type": "error", "message": message})

    async def flush(self):
        events, self._pending = self._pending[:], []
        for ev in events:
            await self._broadcast(ev)

    async def send_event(self, event: dict):
        await self._broadcast(event)

    async def _broadcast(self, event: dict):
        for ws in list(self.clients):
            try:
                await ws.send_json(event)
            except Exception:
                self.clients.discard(ws)


# ===========================================================================
# Turn Record
# ===========================================================================

@dataclass
class TurnRecord:
    turn_number: int
    command: str
    reasoning: str
    game_output: str
    context_snapshot: List[Dict[str, str]]
    stats: Optional[Dict] = None


# ===========================================================================
# Game Manager
# ===========================================================================

class GameManager:
    def __init__(self, story_file: str, config: AgentConfig):
        self.story_file = story_file
        self.config = config
        self.renderer = WebSocketRenderer()
        self.llm = InstrumentedLLMClient(config)
        self.tokenizer = TiktokenTokenizer()
        self.engine: Optional[FrotzEngine] = None
        self.context: Optional[ContextManager] = None

        self.turn_data: List[TurnRecord] = []
        self.command_history: List[str] = []
        self.opening_text: str = ""
        self.current_turn: int = 0
        self.paused: bool = False
        self.stopped: bool = False
        self.rollback_target: Optional[int] = None
        self._cancel_event: Optional[threading.Event] = None
        self._task: Optional[asyncio.Task] = None

    def _init_engine(self):
        self.engine = FrotzEngine(self.story_file)
        self.context = ContextManager(
            self.config, self.llm, self.tokenizer, self.renderer
        )

    async def start(self):
        self._init_engine()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self.stopped = True
        self.paused = False
        self._do_cancel()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self.engine:
            try:
                await asyncio.to_thread(self.engine.close)
            except Exception:
                pass
            self.engine = None

    def _do_cancel(self):
        if self._cancel_event:
            self._cancel_event.set()
        self.llm.cancel_streaming()

    # ------------------------------------------------------------------
    async def do_rollback(self, target_turn: int):
        if self.engine:
            await asyncio.to_thread(self.engine.close)

        commands_to_replay = self.command_history[:target_turn]

        self.engine = FrotzEngine(self.story_file)
        await asyncio.to_thread(self.engine.read_output, 5.0)
        if commands_to_replay:
            await asyncio.to_thread(self.engine.replay_commands, commands_to_replay)

        if 0 < target_turn <= len(self.turn_data):
            self.context.messages = copy.deepcopy(
                self.turn_data[target_turn - 1].context_snapshot
            )
        else:
            self.context.messages = [
                {"role": "system", "content": ContextManager.SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"The game has started. Opening text:\n\n{self.opening_text}"},
            ]

        self.command_history = commands_to_replay
        self.turn_data = self.turn_data[:target_turn]
        self.current_turn = target_turn
        await self.renderer.send_event({"type": "rollback", "to_turn": target_turn})

    # ------------------------------------------------------------------
    async def _stream_llm_turn(self, turn: int) -> str:
        cancel = threading.Event()
        self._cancel_event = cancel
        loop = asyncio.get_event_loop()
        chunk_queue = asyncio.Queue()

        def on_chunk(text):
            loop.call_soon_threadsafe(chunk_queue.put_nowait, text)

        future = loop.run_in_executor(
            None, self.llm.chat_completion_streaming,
            self.context.messages, self.config.max_response_tokens,
            self.config.temperature, cancel, on_chunk,
        )

        first_chunk_time = None
        chunk_count = 0

        while not future.done():
            try:
                text = await asyncio.wait_for(chunk_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            chunk_count += 1
            now = time.monotonic()
            if first_chunk_time is None:
                first_chunk_time = now

            await self.renderer.send_event(
                {"type": "streaming_text", "text": text, "turn": turn}
            )

            # Send live progress every 5 chunks
            if chunk_count % 5 == 0 and first_chunk_time:
                gen_s = now - first_chunk_time
                tps = chunk_count / gen_s if gen_s > 0 else 0
                await self.renderer.send_event({
                    "type": "stream_progress", "turn": turn,
                    "output_tokens": chunk_count,
                    "gen_time_ms": round(gen_s * 1000),
                    "tps": round(tps, 1),
                })

        # Drain remaining
        while not chunk_queue.empty():
            text = chunk_queue.get_nowait()
            chunk_count += 1
            now = time.monotonic()
            if first_chunk_time is None:
                first_chunk_time = now
            await self.renderer.send_event(
                {"type": "streaming_text", "text": text, "turn": turn}
            )

        # Final progress
        if first_chunk_time and chunk_count > 0:
            gen_s = time.monotonic() - first_chunk_time
            tps = chunk_count / gen_s if gen_s > 0 else 0
            await self.renderer.send_event({
                "type": "stream_progress", "turn": turn,
                "output_tokens": chunk_count,
                "gen_time_ms": round(gen_s * 1000),
                "tps": round(tps, 1),
            })

        self._cancel_event = None
        return future.result()

    # ------------------------------------------------------------------
    async def _run_loop(self):
        try:
            opening_raw = await asyncio.to_thread(self.engine.read_output, 5.0)
            self.opening_text = clean_game_output(opening_raw)
            await self.renderer.send_event(
                {"type": "game_output", "text": self.opening_text, "turn": 0}
            )
            self.context.add_user(
                f"The game has started. Opening text:\n\n{self.opening_text}"
            )

            while not self.stopped:
                while self.paused and not self.stopped:
                    if self.rollback_target is not None:
                        target = self.rollback_target
                        self.rollback_target = None
                        await self.do_rollback(target)
                    await asyncio.sleep(0.2)
                if self.stopped:
                    break

                self.current_turn += 1
                turn = self.current_turn
                await self.renderer.send_event({"type": "turn_start", "turn": turn})

                turn_completed = await self._do_turn(turn)
                if not turn_completed:
                    continue

                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            try:
                await self.renderer.send_event(
                    {"type": "error", "message": str(e)}
                )
            except Exception:
                pass
        finally:
            if self.engine:
                try:
                    await asyncio.to_thread(self.engine.close)
                except Exception:
                    pass
                self.engine = None
            await self.renderer.send_event({"type": "session_end"})

    # ------------------------------------------------------------------
    async def _do_turn(self, turn: int) -> bool:
        for attempt in range(1, self.config.llm_retries + 1):
            await asyncio.to_thread(self.context.compact_if_needed)
            await self.renderer.flush()

            # Abort if paused during compaction (hint or pause arrived)
            if self.paused:
                await self.renderer.send_event(
                    {"type": "turn_cancelled", "turn": turn}
                )
                self.current_turn -= 1
                return False

            try:
                full_text = await self._stream_llm_turn(turn)
            except LLMCancelled:
                await self.renderer.send_event(
                    {"type": "turn_cancelled", "turn": turn}
                )
                self.current_turn -= 1
                return False
            except requests.RequestException as e:
                msg = f"Attempt {attempt}/{self.config.llm_retries}: Network error: {e}"
                print(f"[Agent] {msg}", file=sys.stderr, flush=True)
                await self.renderer.send_event(
                    {"type": "turn_retry", "turn": turn, "message": msg}
                )
                await asyncio.sleep(2)
                continue

            result = extract_json_robust(full_text)
            if result and "command" in result:
                reasoning = result.get("reasoning", "No reasoning provided.")
                command = result["command"].strip()
                self.context.add_assistant(reasoning, command)

                # Per-turn stats (directly from API response)
                turn_stats = dict(self.llm.last_stats)

                await self.renderer.send_event({
                    "type": "ai_action",
                    "reasoning": reasoning,
                    "command": command,
                    "turn": turn,
                })
                await self.renderer.send_event({
                    "type": "stats", "turn": turn, **turn_stats,
                })

                if self.rollback_target is not None:
                    target = self.rollback_target
                    self.rollback_target = None
                    self.paused = True
                    await self.renderer.send_event({"type": "paused"})
                    await self.do_rollback(target)
                    return False

                output_raw = await asyncio.to_thread(
                    self.engine.send_command, command
                )
                output_text = clean_game_output(output_raw)
                await self.renderer.send_event(
                    {"type": "game_output", "text": output_text, "turn": turn}
                )

                self.context.add_user(f"Game response:\n\n{output_text}")
                self.command_history.append(command)
                self.turn_data.append(TurnRecord(
                    turn_number=turn,
                    command=command,
                    reasoning=reasoning,
                    game_output=output_text,
                    context_snapshot=copy.deepcopy(self.context.messages),
                    stats=turn_stats,
                ))
                return True
            else:
                preview = full_text[:200].replace('\n', '\\n')
                msg = f"Attempt {attempt}/{self.config.llm_retries}: Invalid JSON. Preview: {preview}"
                print(f"[Agent] {msg}", file=sys.stderr, flush=True)
                await self.renderer.send_event(
                    {"type": "turn_retry", "turn": turn, "message": msg}
                )

        await self.renderer.send_event(
            {"type": "error", "message": "Agent exhausted all retries."}
        )
        self.stopped = True
        return False

    def _translate_hint_sync(self, hint_text: str) -> str:
        messages = [
            {"role": "system", "content": TRANSLATOR_PROMPT},
            {"role": "user", "content": hint_text},
        ]
        return self.llm.chat_completion(messages, 256, 0.3).strip()


# ===========================================================================
# HTTP / WebSocket Handlers
# ===========================================================================

game_manager: Optional[GameManager] = None


async def index_handler(request):
    return web.FileResponse(Path(__file__).parent / "static" / "index.html")


async def games_handler(request):
    games = []
    if GAMES_DIR.exists():
        for f in sorted(GAMES_DIR.iterdir()):
            if f.suffix in ('.z3', '.z4', '.z5', '.z6', '.z7', '.z8', '.zblorb'):
                games.append(f.name)
    return web.json_response(games)


async def start_handler(request):
    global game_manager
    data = await request.json()
    game_file = data.get("game", "planetfall.z5")
    llm_url = data.get("llm_url", "http://localhost:8080/v1/chat/completions")
    model = data.get("model", "gemma-4-31B-it-UD-Q4_K_XL.gguf")

    if game_manager:
        await game_manager.stop()

    config = AgentConfig(llm_url=llm_url, model=model)
    game_manager = GameManager(str(GAMES_DIR / game_file), config)
    await game_manager.start()
    return web.json_response({"status": "started", "game": game_file})


async def websocket_handler(request):
    global game_manager
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    if game_manager:
        game_manager.renderer.clients.add(ws)
        state = {
            "type": "full_state",
            "opening_text": game_manager.opening_text,
            "turns": [
                {
                    "turn": td.turn_number,
                    "reasoning": td.reasoning,
                    "command": td.command,
                    "game_output": td.game_output,
                    "stats": td.stats,
                }
                for td in game_manager.turn_data
            ],
            "paused": game_manager.paused,
        }
        await ws.send_json(state)

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if not game_manager:
                continue

            t = data.get("type")

            if t == "hint":
                was_paused = game_manager.paused
                # Immediately halt any in-flight turn
                if not was_paused:
                    game_manager.paused = True
                    game_manager._do_cancel()
                    await game_manager.renderer.send_event({"type": "paused"})
                # Translate if requested
                hint_text = data.get("text", "")
                if data.get("translate"):
                    try:
                        hint_text = await asyncio.to_thread(
                            game_manager._translate_hint_sync, hint_text
                        )
                    except Exception:
                        pass
                # Inject into context and display
                injected = f"{HINT_PREFIX}: {hint_text}"
                game_manager.context.add_user(injected)
                await game_manager.renderer.send_event({
                    "type": "hint_injected", "text": injected,
                })
                # Auto-resume if game was running
                if not was_paused:
                    game_manager.paused = False
                    await game_manager.renderer.send_event({"type": "resumed"})

            elif t == "pause":
                game_manager.paused = True
                game_manager._do_cancel()
                await game_manager.renderer.send_event({"type": "paused"})
            elif t == "resume":
                game_manager.paused = False
                await game_manager.renderer.send_event({"type": "resumed"})
            elif t == "rollback":
                target = data.get("to_turn", 0)
                if not game_manager.paused:
                    game_manager.paused = True
                    game_manager._do_cancel()
                    await game_manager.renderer.send_event({"type": "paused"})
                game_manager.rollback_target = target
            elif t == "stop":
                await game_manager.stop()
                game_manager = None

        elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
            break

    if game_manager:
        game_manager.renderer.clients.discard(ws)
    return ws


def create_app():
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/games", games_handler)
    app.router.add_post("/start", start_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Adventure Player — Web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    web.run_app(create_app(), host=args.host, port=args.port)
