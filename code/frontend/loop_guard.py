"""Tool-aware no-progress guard for one logical agent turn."""
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import os
import time

__all__ = ["LoopDecision", "LoopGuard"]


@dataclass(frozen=True)
class LoopDecision:
    allow: bool
    code: str = ""
    message: str = ""


@dataclass
class _State:
    turn_key: tuple = ()
    seen: set = field(default_factory=set)
    paths: dict = field(default_factory=dict)
    noop: int = 0
    read_key: tuple = ()
    poll_started: float = 0.0
    unknown_key: tuple = ()
    unknown_started: float = 0.0


class LoopGuard:
    """One public entry: check(session_id, messages) -> LoopDecision."""
    poll_timeout = 60.0
    max_sessions = 256

    def __init__(self):
        self._sessions = OrderedDict()

    def clear(self, session_id):
        self._sessions.pop(session_id, None)

    def check(self, session_id, messages, now=None):
        now = time.monotonic() if now is None else now
        state = self._sessions.get(session_id)
        if state is None:
            state = self._sessions[session_id] = _State()
        else:
            self._sessions.move_to_end(session_id)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

        turn = self._turn(messages)
        if turn is None:
            return LoopDecision(True)
        start, key = turn
        if state.turn_key != key:
            state = self._sessions[session_id] = _State(turn_key=key)

        for call, result in self._new_steps(messages, start, state.seen):
            effect = self._effect(call)
            action = self._action(call)
            observation = self._digest(result.get("content"))

            if effect == "write":
                path = self._path(call)
                version = self._version(path)
                previous = state.paths.get(path)
                state.paths[path] = version
                if previous is not None and version == previous:
                    state.noop += 1
                    if state.noop >= 2:
                        return LoopDecision(False, "no_progress",
                                            "the same write/edit succeeded twice without changing the workspace")
                else:
                    state.noop = 0
            elif effect == "read":
                decision = self._poll(state, action, observation, now, "read")
                if decision: return decision
            else:
                decision = self._poll(state, action, observation, now, "unknown")
                if decision: return decision
        return LoopDecision(True)

    @staticmethod
    def _turn(messages):
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.get("role") == "user":
                text = LoopGuard._text(message.get("content"))
                return index, (index, hashlib.sha256(text.encode()).hexdigest()[:16])
        return None

    @staticmethod
    def _new_steps(messages, start, seen):
        pending = {}
        for message in messages[start:]:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    pending[call.get("id")] = call
            elif message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                call = pending.pop(call_id, None)
                if call is not None and call_id not in seen:
                    seen.add(call_id)
                    yield call, message

    @staticmethod
    def _effect(call):
        name = (call.get("function") or {}).get("name", "")
        return "write" if name in ("write", "edit") else "read" if name in (
            "read", "ls", "cat", "grep", "glob", "stat") else "unknown"

    @staticmethod
    def _action(call):
        function = call.get("function") or {}
        arguments = function.get("arguments", "")
        try:
            arguments = json.dumps(json.loads(arguments), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            pass
        return hashlib.sha256((str(function.get("name")) + "\0" + str(arguments)).encode()).hexdigest()[:16]

    @staticmethod
    def _path(call):
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments", "{}"))
        except (TypeError, ValueError):
            return ""
        for key in ("path", "file_path", "file"):
            value = arguments.get(key)
            if isinstance(value, str) and os.path.isabs(value):
                return value
        return ""

    @staticmethod
    def _version(path):
        if not path:
            return ""
        try:
            stat = os.stat(path)
        except OSError:
            return "missing"
        if stat.st_size <= 2_000_000:
            digest = hashlib.sha256()
            with open(path, "rb") as source:
                for chunk in iter(lambda: source.read(65536), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        return str(stat.st_mtime_ns) + ":" + str(stat.st_size)

    @staticmethod
    def _poll(state, action, observation, now, kind):
        key = (action, observation)
        if kind == "read":
            previous, started = state.read_key, state.poll_started
        else:
            previous, started = state.unknown_key, state.unknown_started
        if key != previous:
            if kind == "read":
                state.read_key = key
                state.poll_started = 0.0
            else:
                state.unknown_key = key
                state.unknown_started = 0.0
            return None
        if not started:
            if kind == "read":
                state.poll_started = now
            else:
                state.unknown_started = now
            return None
        if now - started >= LoopGuard.poll_timeout:
            return LoopDecision(False, "poll_timeout", "agent repeated an unchanged observation past its deadline")
        return None

    @staticmethod
    def _text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return ""

    @staticmethod
    def _digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]
