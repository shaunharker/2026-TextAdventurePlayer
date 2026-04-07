#!/usr/bin/env python3
"""
AI-powered text adventure player.

Refactored for maintainability, testability, and robustness.
Features:
- Proper Dependency Injection (LLM, Tokenizer, Renderer)
- Presentation layer abstraction (Console & HTML rendering)
- Stack-based JSON extraction
- Raw byte-buffer I/O to prevent UTF-8 boundary corruption
- Pagination (*** MORE ***) handling
"""

import argparse
import html as html_mod
import json
import os
import pty
import re
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Protocol
import requests
import tiktoken


# ===========================================================================
# Configuration & Interfaces
# ===========================================================================

@dataclass
class AgentConfig:
    llm_url: str = "http://localhost:8080/v1/chat/completions"
    model: str = "gemma-4-31B-it-UD-Q4_K_XL.gguf"
    context_window: int = 49152
    compact_at_fraction: float = 0.75
    recent_context_tokens: int = 16384
    max_response_tokens: int = 4096
    max_summary_tokens: int = 8192
    llm_retries: int = 10
    temperature: float = 0.7
    raw_temp: float = 0.3
    timeout: int = 240


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...


class LLMProvider(Protocol):
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str: ...


class OutputRenderer(Protocol):
    """Abstracts the presentation layer from the game logic."""
    def session_start(self) -> None: ...
    def game_output(self, text: str) -> None: ...
    def turn_start(self, turn: int) -> None: ...
    def ai_action(self, reasoning: str, command: str) -> None: ...
    def system_message(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...
    def session_end(self) -> None: ...


# ===========================================================================
# Output Renderers
# ===========================================================================

class ConsoleRenderer:
    """Standard stdout text renderer."""
    def session_start(self):
        print("=== AI Adventure Player Started ===\n")

    def game_output(self, text: str):
        print(f"{text}\n")

    def turn_start(self, turn: int):
        print(f"--- Turn {turn} ---")

    def ai_action(self, reasoning: str, command: str):
        print(f"[AI Thoughts]  {reasoning}")
        print(f"[AI Command]   > {command}\n")

    def system_message(self, message: str):
        print(f"[System] {message}")

    def error(self, message: str):
        print(f"[Error] {message}", file=sys.stderr)

    def session_end(self):
        print("=== Game Session Terminated ===")


class HTMLRenderer:
    """Renders the game session to an HTML file using a card-based layout."""

    _HTML_HEADER = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Adventure Player Transcript</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --turn-bg: #0b1120;
            --border-color: #334155;
            --text-main: #cbd5e1;
            --text-reasoning: #a78bfa;
            --text-command: #4ade80;
            --text-response: #f8fafc;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: var(--font-sans);
            margin: 0;
            padding: 40px 20px;
            line-height: 1.6;
        }
        .container { max-width: 950px; margin: 0 auto; }
        .turn-card {
            display: flex;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 24px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        .turn-sidebar {
            flex: 0 0 100px;
            background: var(--turn-bg);
            border-right: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: flex-start;
            flex-direction: column;
            padding: 24px 10px;
        }
        .turn-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
        }
        .turn-number {
            font-size: 2.2rem;
            font-weight: 700;
            color: #e2e8f0;
            line-height: 1;
            margin-top: 8px;
        }
        .turn-content { flex: 1; padding: 24px; min-width: 0; }
        .reasoning {
            color: var(--text-reasoning);
            font-style: italic;
            margin-bottom: 20px;
            font-size: 1.05rem;
        }
        .command {
            font-family: var(--font-mono);
            color: var(--text-command);
            margin-bottom: 20px;
            font-size: 1.1rem;
            background: rgba(15, 23, 42, 0.6);
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 4px solid var(--text-command);
        }
        .command strong { font-weight: 700; }
        .response {
            font-family: var(--font-mono);
            color: var(--text-response);
            white-space: pre-wrap;
            word-wrap: break-word;
            background: #0f172a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            font-size: 0.95rem;
        }
        .response::-webkit-scrollbar { height: 8px; }
        .response::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
"""

    def __init__(self, filepath: str = "session.html"):
        self.filepath = filepath
        self._current_turn = -1  # no turn yet
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(self._HTML_HEADER)

    def session_start(self):
        pass  # header already written in __init__

    def game_output(self, text: str):
        escaped = html_mod.escape(text)
        if self._current_turn <= 0:
            # Turn 0: opening text, wrap in its own card
            self._write(
                '        <div class="turn-card">\n'
                '            <div class="turn-sidebar">\n'
                '                <div class="turn-label">Turn</div>\n'
                '                <div class="turn-number">0</div>\n'
                '            </div>\n'
                '            <div class="turn-content">\n'
                f'                <div class="response">{escaped}</div>\n'
                '            </div>\n'
                '        </div>\n\n'
            )
        else:
            # Response for an active turn card — write response and close card
            self._write(
                f'                <div class="response">{escaped}</div>\n'
                '            </div>\n'
                '        </div>\n\n'
            )

    def turn_start(self, turn: int):
        self._current_turn = turn
        self._write(
            '        <div class="turn-card">\n'
            '            <div class="turn-sidebar">\n'
            '                <div class="turn-label">Turn</div>\n'
            f'                <div class="turn-number">{turn}</div>\n'
            '            </div>\n'
            '            <div class="turn-content">\n'
        )

    def ai_action(self, reasoning: str, command: str):
        r_esc = html_mod.escape(reasoning)
        c_esc = html_mod.escape(command)
        self._write(
            f'                <div class="reasoning">{r_esc}</div>\n'
            f'                <div class="command">&gt; <strong>{c_esc}</strong></div>\n'
        )

    def system_message(self, message: str):
        pass  # system messages are console-only

    def error(self, message: str):
        pass  # errors are console-only

    def session_end(self):
        self._write("    </div>\n</body>\n</html>\n")

    def _write(self, content: str):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(content)


class MultiRenderer:
    """Dispatches to multiple renderers so stdout and HTML run simultaneously."""
    def __init__(self, renderers: List):
        self.renderers = renderers

    def session_start(self):
        for r in self.renderers:
            r.session_start()

    def game_output(self, text: str):
        for r in self.renderers:
            r.game_output(text)

    def turn_start(self, turn: int):
        for r in self.renderers:
            r.turn_start(turn)

    def ai_action(self, reasoning: str, command: str):
        for r in self.renderers:
            r.ai_action(reasoning, command)

    def system_message(self, message: str):
        for r in self.renderers:
            r.system_message(message)

    def error(self, message: str):
        for r in self.renderers:
            r.error(message)

    def session_end(self):
        for r in self.renderers:
            r.session_end()


# ===========================================================================
# Implementations
# ===========================================================================

class TiktokenTokenizer:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))


class OpenAILLMClient:
    def __init__(self, config: AgentConfig):
        self.config = config

    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(self.config.llm_url, json=payload, timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ===========================================================================
# Utilities
# ===========================================================================

def extract_json_robust(text: str) -> Optional[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    idx = 0
    while idx < len(text):
        start_idx = text.find('{', idx)
        if start_idx == -1:
            return None

        stack = []
        found_end = False
        for i in range(start_idx, len(text)):
            char = text[i]
            if char == '{':
                stack.append('{')
            elif char == '}':
                if stack:
                    stack.pop()
                if not stack:
                    json_str = text[start_idx:i+1]
                    try:
                        obj = json.loads(json_str)
                        if isinstance(obj, dict) and "command" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    idx = i + 1
                    found_end = True
                    break
        if not found_end:
            return None

    return None

def clean_game_output(text: str) -> str:
    text = re.sub(r"^Using normal formatting\..*?\n", "", text)
    text = re.sub(r"^Loading .*?\.\n+", "", text)
    text = re.sub(r"\n?>[ \t]*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ===========================================================================
# Engine Management
# ===========================================================================

class FrotzEngine:
    def __init__(self, story_file: str):
        self.master_fd, slave_fd = pty.openpty()
        # Plain Ascii: -p
        # no "**more**" prompts: -m
        # Random seed 0: -s 0
        # fixed width at 60: -w 60
        self.proc = subprocess.Popen(["dfrotz", "-p", "-m", "-s", "0", "-w", "60", story_file],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)

    def read_output(self, timeout: float = 5.0) -> str:
        buffer = bytearray()
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            ready, _, _ = select.select([self.master_fd], [], [], min(remaining, 0.1))
            if ready:
                try:
                    chunk = os.read(self.master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break

                buffer.extend(chunk)
                text_so_far = buffer.decode("utf-8", errors="replace")

                if re.search(r"\*\*\*[ \t]*MORE[ \t]*\*\*\*", text_so_far, re.IGNORECASE):
                    os.write(self.master_fd, b"\n")
                    continue
            elif buffer:
                # No data ready and we have output — frotz has paused writing.
                # Check for the input prompt now (safe from intermediate matches).
                text_so_far = buffer.decode("utf-8", errors="replace")
                if re.search(r"\n>[ \t]*$", text_so_far):
                    break

        return buffer.decode("utf-8", errors="replace")

    def replay_commands(self, commands: list, timeout: float = 10.0):
        """Blast all commands at once and drain output — instant replay."""
        if not commands:
            return
        blob = "\n".join(commands) + "\n"
        os.write(self.master_fd, blob.encode("utf-8"))
        # Drain output until frotz is idle (no data ready + prompt visible)
        buf = bytearray()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([self.master_fd], [], [], min(remaining, 0.05))
            if ready:
                try:
                    chunk = os.read(self.master_fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf.extend(chunk)
            elif buf:
                # No more data flowing — check for the final prompt
                text = buf.decode("utf-8", errors="replace")
                if re.search(r"\n>[ \t]*$", text):
                    break

    def send_command(self, command: str) -> str:
        os.write(self.master_fd, (command + "\n").encode("utf-8"))
        output = self.read_output()
        lines = output.split("\n")
        if lines and lines[0].strip() == command.strip():
            output = "\n".join(lines[1:])
        return output

    def close(self):
        try:
            os.write(self.master_fd, b"quit\n")
            time.sleep(0.2)
            os.write(self.master_fd, b"y\n")
        except OSError:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        try:
            os.close(self.master_fd)
        except OSError:
            pass


# ===========================================================================
# Context & State Management
# ===========================================================================

class ContextManager:
    SYSTEM_PROMPT = """\
You are an interactive fiction player. You are playing a text adventure \
game powered by a Z-machine interpreter. Your goal is to explore, solve puzzles, \
and make progress through the story.

Rules:
- Respond with a JSON object.
- Format: {"reasoning": "<your thoughts>", "command": "<your next game command>", }
- "reasoning" field:
  - think of this as jotting down a few lines into your notebook of thoughts
  - adopt a spirited persona, making unnecessarily poetic comments about your situation
  - reason out loud to yourself
  - think critically about whether what you are doing is productive or repetitive and bloody-minded
  - make plans
  - explain the goal of the command you will take
  - occasionally remind yourself of these habits of thoughts so they continue to appear throughout your context
- "command" field:
  - this is the literal instruction you send to the text adventure parser
  - Examine things, map your surroundings, try reasonable actions.
  - Keep commands short and direct (1-4 words typically).
  - Some standard interactive fiction commands: look, i, n, w, s, e, up, down, sw, ne, sw, se, examine, wait

Hints and Tips:
- Use 'look' and 'examine' to inspect your surroundings
- Don't go into an infinite wait loop just because you are stuck
- Be careful not to miss map connections which might be implicit.
  - For example, if you are in an east-west hallway, then east and west are likely directions you can go.
- Don't try to brute force puzzles (e.g. don't try every combination on a lock, actually play the game)
"""

    SUMMARY_PROMPT = """\
You are summarizing a text adventure game session for an AI player that needs \
to continue playing. The player's conversation history is getting too long and \
must be compressed.

Produce a concise but thorough summary covering:
1. STORY SO FAR: Key plot events and narrative progress
2. CURRENT STATE: Where the player is right now, what just happened
3. INVENTORY: All items the player is carrying
4. MAP KNOWLEDGE: Locations discovered and how they connect
5. PUZZLES & GOALS: What the player is trying to do, what's been solved, what's blocked
6. HINTS: Anything the player has learned that might be useful later (clues, \
   locked doors, NPCs, items seen but not taken, failed attempts)
7. WALKTHROUGH: (Important!) Summarize the steps required to resolve the game from a restarted state.

Be factual and specific. Do not invent details not present in the history.\
"""

    def __init__(self, config: AgentConfig, llm: LLMProvider, tokenizer: Tokenizer, renderer: OutputRenderer):
        self.config = config
        self.llm = llm
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.messages: List[Dict[str, str]] =[{"role": "system", "content": self.SYSTEM_PROMPT}]

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, reasoning: str, command: str):
        payload = json.dumps({"reasoning": reasoning, "command": command})
        self.messages.append({"role": "assistant", "content": payload})

    def get_total_tokens(self) -> int:
        return sum(self.tokenizer.count_tokens(m["content"]) for m in self.messages)

    def compact_if_needed(self) -> bool:
        current_tokens = self.get_total_tokens()
        threshold = int(self.config.context_window * self.config.compact_at_fraction)
        
        if current_tokens < threshold:
            return False

        self.renderer.system_message(f"Tokens at {current_tokens} / {self.config.context_window}. Compacting history...")
        self._perform_compaction()
        return True

    def _perform_compaction(self):
        transcript_parts = []
        for msg in self.messages[1:]:
            prefix = "GAME: " if msg["role"] == "user" else "PLAYER: "
            transcript_parts.append(f"{prefix}{msg['content']}")
        
        transcript = "\n\n".join(transcript_parts)
        summary_messages =[
            {"role": "system", "content": self.SUMMARY_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{transcript}"}
        ]

        try:
            summary = self.llm.chat_completion(
                summary_messages, 
                max_tokens=self.config.max_summary_tokens, 
                temperature=self.config.raw_temp
            )
        except Exception as e:
            self.renderer.error(f"Summary generation failed: {e}. Falling back to truncation.")
            self.messages =[self.messages[0]] + self.messages[-10:]
            return

        enhanced_system = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"--- EARLIER GAMEPLAY SUMMARY ---\n{summary}\n--- END SUMMARY ---\n"
        )

        recent =[]
        tokens_used = 0
        budget = self.config.recent_context_tokens

        for msg in reversed(self.messages[1:]):
            msg_tokens = self.tokenizer.count_tokens(msg["content"])
            if tokens_used + msg_tokens > budget and recent:
                break
            recent.insert(0, msg)
            tokens_used += msg_tokens

        self.messages =[{"role": "system", "content": enhanced_system}] + recent


# ===========================================================================
# Agent Core
# ===========================================================================

class Agent:
    def __init__(self, config: AgentConfig, llm: LLMProvider, context: ContextManager, renderer: OutputRenderer):
        self.config = config
        self.llm = llm
        self.context = context
        self.renderer = renderer

    def get_next_action(self) -> tuple[str, str]:
        for attempt in range(1, self.config.llm_retries + 1):
            try:
                self.context.compact_if_needed()
                
                raw_response = self.llm.chat_completion(
                    self.context.messages,
                    max_tokens=self.config.max_response_tokens,
                    temperature=self.config.temperature
                )
                
                result = extract_json_robust(raw_response)
                if result and "command" in result:
                    reasoning = result.get("reasoning", "No reasoning provided.")
                    command = result["command"].strip()
                    self.context.add_assistant(reasoning, command)
                    return reasoning, command

                preview = raw_response[:200].replace('\n', '\\n')
                msg = f"Attempt {attempt}/{self.config.llm_retries}: Invalid JSON. Preview: {preview}"
                print(f"[Agent] {msg}", file=sys.stderr, flush=True)
                self.renderer.system_message(msg)
            except requests.RequestException as e:
                msg = f"Attempt {attempt}/{self.config.llm_retries}: Network error: {e}"
                print(f"[Agent] {msg}", file=sys.stderr, flush=True)
                self.renderer.error(msg)
                time.sleep(2)
                
        raise RuntimeError("Agent exhausted all retries attempting to generate a valid action.")


# ===========================================================================
# Main Game Loop Orchestrator
# ===========================================================================

class GameSession:
    def __init__(self, engine: FrotzEngine, agent: Agent, renderer: OutputRenderer):
        self.engine = engine
        self.agent = agent
        self.renderer = renderer

    def run(self, max_turns: int):
        self.renderer.session_start()
        try:
            opening_raw = self.engine.read_output(timeout=5.0)
            opening_text = clean_game_output(opening_raw)
            self.renderer.game_output(opening_text)
            
            self.agent.context.add_user(f"The game has started. Opening text:\n\n{opening_text}")

            for turn in range(1, max_turns + 1):
                self.renderer.turn_start(turn)
                
                reasoning, command = self.agent.get_next_action()
                self.renderer.ai_action(reasoning, command)

                output_raw = self.engine.send_command(command)
                output_text = clean_game_output(output_raw)
                self.renderer.game_output(output_text)
                
                self.agent.context.add_user(f"Game response:\n\n{output_text}")
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            self.renderer.system_message("Session Interrupted by User")
        except Exception as e:
            self.renderer.error(f"Session Fault: {e}")
        finally:
            self.engine.close()
            self.renderer.session_end()


# ===========================================================================
# Entry Point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="AI-powered text adventure player")
    parser.add_argument("story", help="Path to Z-machine story file")
    parser.add_argument("-n", "--max-turns", type=int, default=9999)
    parser.add_argument("--html-file", default="session.html", help="Path for HTML transcript (default: session.html)")
    parser.add_argument("--no-html", action="store_true", help="Disable HTML output")
    args = parser.parse_args()

    config = AgentConfig()
    tokenizer = TiktokenTokenizer()
    llm = OpenAILLMClient(config)

    renderers = [ConsoleRenderer()]
    if not args.no_html:
        renderers.append(HTMLRenderer(filepath=args.html_file))
    renderer = MultiRenderer(renderers)

    context = ContextManager(config, llm, tokenizer, renderer)
    agent = Agent(config, llm, context, renderer)
    engine = FrotzEngine(args.story)

    session = GameSession(engine, agent, renderer)
    session.run(args.max_turns)


if __name__ == "__main__":
    main()
