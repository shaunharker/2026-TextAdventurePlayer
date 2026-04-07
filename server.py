#!/usr/bin/env python3
"""
Web-based AI text adventure player.
Serves a browser UI for watching an AI play interactive fiction,
with live updates via WebSocket, hint injection, and rollback support.
"""

import asyncio
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import requests
from aiohttp import web

from main import (
    AgentConfig, FrotzEngine, ContextManager, Agent,
    TiktokenTokenizer, clean_game_output,
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
# Instrumented LLM Client  (captures usage / timing per call)
# ===========================================================================

class InstrumentedLLMClient:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.last_stats: Dict = {}

    def chat_completion(self, messages, max_tokens, temperature) -> str:
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
            "llm_time_ms": round(elapsed_ms),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached,
            "uncached_tokens": prompt_tokens - cached,
            "output_tokens": output_tokens,
        }
        return data["choices"][0]["message"]["content"]


# ===========================================================================
# WebSocket Renderer  (satisfies the OutputRenderer protocol)
# ===========================================================================

class WebSocketRenderer:
    """Sync methods queue events; async flush/send_event broadcast them."""

    def __init__(self):
        self.clients: set = set()
        self._pending: List[dict] = []

    # --- sync (called by Agent / ContextManager in worker thread) ----------
    def session_start(self): pass
    def game_output(self, text: str): pass
    def turn_start(self, turn: int): pass
    def ai_action(self, reasoning: str, command: str): pass
    def session_end(self): pass

    def system_message(self, message: str):
        self._pending.append({"type": "system_message", "message": message})

    def error(self, message: str):
        self._pending.append({"type": "error", "message": message})

    # --- async (called by game loop) --------------------------------------
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
# Turn Record  (for rollback)
# ===========================================================================

@dataclass
class TurnRecord:
    turn_number: int
    command: str
    reasoning: str
    game_output: str
    context_snapshot: List[Dict[str, str]]


# ===========================================================================
# Game Manager  (owns engine + agent + game loop)
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
        self.agent: Optional[Agent] = None

        self.turn_data: List[TurnRecord] = []
        self.command_history: List[str] = []
        self.opening_text: str = ""
        self.current_turn: int = 0

        self.paused: bool = False
        self.stopped: bool = False
        self.rollback_target: Optional[int] = None
        self.hint_queue: asyncio.Queue = asyncio.Queue()

        self.total_uncached: int = 0
        self.total_output: int = 0
        self.total_llm_time: int = 0
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    def _init_engine(self):
        self.engine = FrotzEngine(self.story_file)
        self.context = ContextManager(
            self.config, self.llm, self.tokenizer, self.renderer
        )
        self.agent = Agent(self.config, self.llm, self.context, self.renderer)

    async def start(self):
        self._init_engine()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self.stopped = True
        self.paused = False
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

    # ------------------------------------------------------------------
    async def do_rollback(self, target_turn: int):
        if self.engine:
            await asyncio.to_thread(self.engine.close)

        commands_to_replay = self.command_history[:target_turn]

        # Fresh engine — deterministic because dfrotz uses -s 0
        self.engine = FrotzEngine(self.story_file)
        await asyncio.to_thread(self.engine.read_output, 5.0)
        # Blast all replay commands at once (like piping to frotz stdin)
        if commands_to_replay:
            await asyncio.to_thread(self.engine.replay_commands, commands_to_replay)

        # Restore context
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
        self.agent = Agent(self.config, self.llm, self.context, self.renderer)
        await self.renderer.send_event({"type": "rollback", "to_turn": target_turn})

    # ------------------------------------------------------------------
    async def _run_loop(self):
        try:
            # Opening text
            opening_raw = await asyncio.to_thread(self.engine.read_output, 5.0)
            self.opening_text = clean_game_output(opening_raw)
            await self.renderer.send_event(
                {"type": "game_output", "text": self.opening_text, "turn": 0}
            )
            self.context.add_user(
                f"The game has started. Opening text:\n\n{self.opening_text}"
            )

            while not self.stopped:
                # ---- pause loop (with rollback check) ----
                while self.paused and not self.stopped:
                    if self.rollback_target is not None:
                        target = self.rollback_target
                        self.rollback_target = None
                        await self.do_rollback(target)
                    await asyncio.sleep(0.2)
                if self.stopped:
                    break

                # ---- process queued hints ----
                while not self.hint_queue.empty():
                    hint_data = self.hint_queue.get_nowait()
                    hint_text = hint_data["text"]
                    original = hint_text
                    if hint_data.get("translate"):
                        try:
                            hint_text = await asyncio.to_thread(
                                self._translate_hint_sync, hint_text
                            )
                        except Exception:
                            pass
                    self.context.add_user(f"{HINT_PREFIX}: {hint_text}")
                    await self.renderer.send_event({
                        "type": "hint_injected",
                        "original": original,
                        "translated": hint_text,
                    })

                # ---- new turn ----
                self.current_turn += 1
                turn = self.current_turn
                await self.renderer.send_event({"type": "turn_start", "turn": turn})

                # LLM action (blocking → thread)
                try:
                    reasoning, command = await asyncio.to_thread(
                        self.agent.get_next_action
                    )
                except RuntimeError as e:
                    await self.renderer.send_event(
                        {"type": "error", "message": str(e)}
                    )
                    break

                # Flush system_message / error events emitted by Agent
                await self.renderer.flush()

                # Rollback requested while LLM was thinking?
                if self.rollback_target is not None:
                    target = self.rollback_target
                    self.rollback_target = None
                    self.paused = True
                    await self.renderer.send_event({"type": "paused"})
                    await self.do_rollback(target)
                    continue

                # Stats
                stats = dict(self.llm.last_stats)
                self.total_uncached += stats.get("uncached_tokens", 0)
                self.total_output += stats.get("output_tokens", 0)
                self.total_llm_time += stats.get("llm_time_ms", 0)
                stats["total_uncached"] = self.total_uncached
                stats["total_output"] = self.total_output
                stats["total_llm_time"] = self.total_llm_time

                await self.renderer.send_event({
                    "type": "ai_action",
                    "reasoning": reasoning,
                    "command": command,
                    "turn": turn,
                })
                await self.renderer.send_event({"type": "stats", **stats})

                # Frotz
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
                ))
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
        # Catch-up: send full state
        state = {
            "type": "full_state",
            "opening_text": game_manager.opening_text,
            "turns": [
                {
                    "turn": td.turn_number,
                    "reasoning": td.reasoning,
                    "command": td.command,
                    "game_output": td.game_output,
                }
                for td in game_manager.turn_data
            ],
            "paused": game_manager.paused,
            "stats": {
                **game_manager.llm.last_stats,
                "total_uncached": game_manager.total_uncached,
                "total_output": game_manager.total_output,
                "total_llm_time": game_manager.total_llm_time,
            } if game_manager.llm.last_stats else {},
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
                await game_manager.hint_queue.put({
                    "text": data.get("text", ""),
                    "translate": data.get("translate", False),
                })

            elif t == "pause":
                game_manager.paused = True
                await game_manager.renderer.send_event({"type": "paused"})

            elif t == "resume":
                game_manager.paused = False
                await game_manager.renderer.send_event({"type": "resumed"})

            elif t == "rollback":
                target = data.get("to_turn", 0)
                if not game_manager.paused:
                    game_manager.paused = True
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


# ===========================================================================
# App
# ===========================================================================

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
