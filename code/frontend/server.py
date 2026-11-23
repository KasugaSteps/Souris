"""Serial-backend OpenAI adapter with shared in-flight requests and SSE replay."""
import argparse
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import time
import uuid

from fastapi import FastAPI, Request
from starlette.exceptions import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import jsonschema
if __package__:
    from .images import ImageError, PLACEHOLDER, validate_content, prepare_images
    from .tools import GatewayRetry, ToolCallAssembler, ToolProtocolError, truncate_tool_messages
    from .errors import ApiError, render_http, render_sse
    from .loop_guard import LoopGuard
else:
    from images import ImageError, PLACEHOLDER, validate_content, prepare_images
    from tools import GatewayRetry, ToolCallAssembler, ToolProtocolError, truncate_tool_messages
    from errors import ApiError, render_http, render_sse
    from loop_guard import LoopGuard



def content_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        try:
            validate_content(content)
        except ImageError as exc:
            raise ApiError(str(exc), "image_invalid") from exc
        return "".join(p["text"] if p["type"] == "text" else PLACEHOLDER for p in content)
    raise ApiError("invalid message content")


def validate(body, chat):
    if not isinstance(body, dict):
        raise ApiError("expected a JSON object")
    allowed = {"model", "max_tokens", "temperature", "top_p", "top_k", "min_p", "stop", "seed", "stream", "stream_options", "n"}
    allowed |= {"messages", "tools", "tool_choice", "parallel_tool_calls", "max_calls", "response_format", "reasoning_effort"} if chat else {"prompt"}
    unknown = set(body) - allowed
    if unknown:
        raise ApiError("unsupported field: " + sorted(unknown)[0])
    if "model" in body and not isinstance(body["model"], str):
        raise ApiError("model must be a string")
    if body.get("n", 1) != 1 or isinstance(body.get("n"), bool):
        raise ApiError("only n=1 is supported")
    count = body.get("max_tokens", 128)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ApiError("max_tokens must be a positive integer")
    for key, default in (("temperature", 0), ("top_p", 1)):
        value = body.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ApiError("invalid " + key)
        if value < 0 if key == "temperature" else not 0 < value <= 1:
            raise ApiError("invalid " + key)
    top_k, min_p = body.get("top_k", 0), body.get("min_p", 0)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ApiError("top_k must be a nonnegative integer")
    if isinstance(min_p, bool) or not isinstance(min_p, (int, float)) or not math.isfinite(min_p) or not 0 <= min_p <= 1:
        raise ApiError("min_p must be in [0, 1]")
    if "seed" in body and (isinstance(body["seed"], bool) or not isinstance(body["seed"], int) or body["seed"] < 0 or body["seed"] >= 2**64):
        raise ApiError("seed must be a uint64 integer")
    for key in ("stream", "parallel_tool_calls"):
        if key in body and not isinstance(body[key], bool):
            raise ApiError(key + " must be boolean")
    maximum = body.get("max_calls", 4)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 8:
        raise ApiError("max_calls must be in 1..8")
    stream_options = body.get("stream_options", {})
    if not isinstance(stream_options, dict) or set(stream_options) - {"include_usage"} or not isinstance(stream_options.get("include_usage", False), bool):
        raise ApiError("invalid stream_options")
    stops = body.get("stop")
    if stops is not None:
        stops = [stops] if isinstance(stops, str) else stops
        if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
            raise ApiError("stop must contain nonempty strings")
    if chat:
        if body.get("reasoning_effort", "none") not in ("none", "minimal", "low", "medium", "high"):
            raise ApiError("invalid reasoning_effort")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ApiError("messages must be a nonempty array")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant", "tool"):
                raise ApiError("unsupported message role")
            if set(message) - {"role", "content", "name", "tool_calls", "tool_call_id", "reasoning_content"}:
                raise ApiError("unsupported message field")
            if "reasoning_content" in message and (message["role"] != "assistant" or not isinstance(message["reasoning_content"], str)):
                raise ApiError("reasoning_content must be an assistant string")
            try:
                validate_content(message.get("content"))
            except ImageError as exc:
                raise ApiError(str(exc), "image_invalid") from exc
        if "response_format" in body and body["response_format"] not in ({"type": "text"}, {"type": "json_object"}):
            raise ApiError("only text or prompt-hinted json_object response_format is supported")
    elif not isinstance(body.get("prompt"), str):
        raise ApiError("prompt must be a string")


def tools_for(body):
    tools = body.get("tools", [])
    if not isinstance(tools, list):
        raise ApiError("tools must be an array")
    functions = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ApiError("only function tools are supported")
        fn = tool["function"]
        if set(fn) - {"name", "description", "parameters", "strict"}:
            raise ApiError("unsupported function field")
        name, schema = fn.get("name"), fn.get("parameters", {"type": "object"})
        if not isinstance(schema, dict):
            raise ApiError("tool parameters must be a JSON schema object")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) or name in functions:
            raise ApiError("invalid or duplicate tool name")
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.SchemaError as exc:
            raise ApiError("invalid tool parameter schema") from exc
        functions[name] = {**fn, "parameters": schema}
    choice = body.get("tool_choice", "auto")
    if isinstance(choice, dict):
        if set(choice) != {"type", "function"} or choice["type"] != "function" or not isinstance(choice["function"], dict):
            raise ApiError("invalid tool_choice")
        name = choice["function"].get("name")
        if name not in functions or set(choice["function"]) != {"name"}:
            raise ApiError("tool_choice names an unavailable function")
    elif choice not in ("auto", "none", "required"):
        raise ApiError("invalid tool_choice")
    if choice == "required" and not functions:
        raise ApiError("tool_choice required needs tools")
    return functions


def tool_constraint_build(body, functions):
    """Only structured schemas cross the raw API; clients cannot submit GBNF."""
    choice = body.get("tool_choice", "auto")
    if not functions or choice == "none":
        return None
    allowed = {"type", "properties", "required", "additionalProperties", "items", "enum", "const", "description", "default"}
    def check(schema):
        if not isinstance(schema, dict) or set(schema) - allowed:
            raise ApiError("unsupported tool schema")
        for sub in schema.get("properties", {}).values():
            check(sub)
        if "items" in schema:
            check(schema["items"])
    for fn in functions.values():
        check(fn["parameters"])
    return {"kind": "tool_call", "choice": {"name": choice["function"]["name"]} if isinstance(choice, dict) else choice,
            "tools": [{key: fn[key] for key in ("name", "description", "parameters") if key in fn} for fn in functions.values()],
            "max_calls": body.get("max_calls", 4) if body.get("parallel_tool_calls", True) else 1,
            "parallel_tool_calls": body.get("parallel_tool_calls", True)}


def chat_template(body, functions):
    """Qwen role delimiters and Hermes JSON tool blocks, with exact assistant reasoning preserved when supplied."""
    systems, turns, outstanding = [], [], {}
    choice = body.get("tool_choice", "auto")
    if functions and choice != "none":
        systems.append("Available functions are described below. When calling a function, write a JSON object with keys name and arguments inside <tool_call> and </tool_call>.\n<tools>\n" +
                       "\n".join(json.dumps(fn, ensure_ascii=False) for fn in functions.values()) + "\n</tools>")
        if choice == "required":
            systems.append("Call at least one of the available functions in this response.")
        elif isinstance(choice, dict):
            systems.append("Call the function " + choice["function"]["name"] + " in this response.")
        if not body.get("parallel_tool_calls", True):
            systems.append("Make at most one function call in this response.")
    if choice == "none":
        systems.append("Answer using text. Do not call functions in this response.")
    if body.get("response_format") == {"type": "json_object"}:
        systems.append("Return a JSON object as the response text.")
    for message in body["messages"]:
        role, text = message["role"], content_text(message.get("content"))
        if outstanding and role != "tool":
            raise ApiError("every assistant tool call needs a tool result before the next turn")
        if role == "system":
            systems.append(text)
            continue
        if role == "assistant":
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list):
                raise ApiError("tool_calls must be an array")
            if calls and not text.strip() and "reasoning_content" not in message:
                text = ""
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function" or not isinstance(call.get("function"), dict):
                    raise ApiError("invalid assistant tool call")
                fn, call_id = call["function"], call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in outstanding or not isinstance(fn.get("name"), str):
                    raise ApiError("invalid tool_call_id")
                try:
                    args = json.loads(fn["arguments"]) if isinstance(fn.get("arguments"), str) else fn["arguments"]
                except (KeyError, ValueError) as exc:
                    raise ApiError("invalid tool arguments") from exc
                if not isinstance(args, dict):
                    raise ApiError("tool arguments must be an object")
                outstanding[call_id] = fn["name"]
                text += ("\n" if text and not text.endswith("\n") else "") + "<tool_call>\n" + json.dumps({"name": fn["name"], "arguments": args}, ensure_ascii=False) + "\n</tool_call>"
            # Reproduce the generation header already consumed by the model;
            # omitting it makes every completed assistant turn break prefix reuse.
            text = ("<think>\n" + message["reasoning_content"] + "\n</think>\n\n" if "reasoning_content" in message
                    else "<think>\n\n</think>\n\n") + text
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in outstanding:
                raise ApiError("tool_call_id has no preceding assistant call")
            del outstanding[call_id]
            role, text = "user", "<tool_response>\n" + text + "\n</tool_response>"
        turns.append("<|im_start|>" + role + "\n" + text + "<|im_end|>\n")
    if outstanding:
        raise ApiError("every assistant tool call needs a tool result before generation")
    effort = body.get("reasoning_effort", "none")
    if effort != "none":
        systems.append(_EFFORT_INSTRUCTIONS[effort])
    prefix = "<|im_start|>system\n" + "\n\n".join(systems) + "<|im_end|>\n" if systems else ""
    header = "<think>\n" if thinking(body) else "<think>\n\n</think>\n\n"
    return prefix + "".join(turns) + "<|im_start|>assistant\n" + header


_EFFORT_INSTRUCTIONS = {
    "minimal": "Reasoning effort: minimal. Use the shortest reasoning needed, then answer.",
    "low": "Reasoning effort: low. Keep reasoning brief, then answer.",
    "medium": "Reasoning effort: medium. Use a moderate amount of reasoning, then answer.",
    "high": "Reasoning effort: high. Reason thoroughly and check the result before answering.",
}


def thinking(body):
    return body.get("reasoning_effort", "none") != "none"


class _ReasoningStream:
    """Split tags across arbitrary SSE boundaries without rewriting thought text."""
    def __init__(self):
        self.pending = ""
        self.start = True
        self.closed = False
        self.reasoning = []
        self.content = []
        self.content_start = True
        self.separator = ""

    def _content(self, text, final=False):
        if self.content_start:
            self.separator += text
            if not final and "\n\n".startswith(self.separator) and len(self.separator) < 2:
                return []
            text = self.separator[2:] if self.separator.startswith("\n\n") else self.separator
            self.separator = ""
            self.content_start = False
        if text:
            self.content.append(text)
            return [("content", text)]
        return []

    def feed(self, text, final=False):
        if self.closed:
            return self._content(text, final)
        self.pending += text
        if self.start:
            opening = "<think>"
            if opening.startswith(self.pending) and len(self.pending) < len(opening):
                return []
            if self.pending.startswith(opening):
                self.pending = self.pending[len(opening):]
            self.start = False
        closing = "</think>"
        end = self.pending.find(closing)
        events = []
        if end >= 0:
            reason, content = self.pending[:end], self.pending[end + len(closing):]
            # Remove only template framing; preserve the reasoning payload verbatim.
            if reason.endswith("\n"):
                reason = reason[:-1]
            self.closed = True
            self.pending = ""
            if reason:
                self.reasoning.append(reason)
                events.append(("reasoning_content", reason))
            events.extend(self._content(content, final))
        else:
            hold = 0
            for size in range(1, min(len(closing), len(self.pending)) + 1):
                if closing.startswith(self.pending[-size:]):
                    hold = size
            reason = self.pending[:-hold] if hold else self.pending
            if reason.endswith("\n"):
                reason = reason[:-1]
                hold += 1
            self.pending = self.pending[-hold:] if hold else ""
            if reason:
                self.reasoning.append(reason)
                events.append(("reasoning_content", reason))
        return events

    def finish(self):
        if not self.closed:
            raise ToolProtocolError("incomplete_reasoning")
        self._content("", final=True)
        return "".join(self.reasoning), "".join(self.content)


@dataclass
class Settings:
    backend: str = "http://127.0.0.1:8000"
    connect: float = 10
    ttft: float = 120
    idle: float = 60
    total: float = 900
    ttl: float = 600
    max_entries: int = 256


@dataclass
class Job:
    digest: str
    request_id: str
    body: dict
    chat: bool
    session_id: str
    events: list = field(default_factory=list)
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task | None = None
    response: dict | None = None
    error: ApiError | None = None
    done: bool = False
    saw_delta: bool = False
    subscribers: int = 0
    finished: float = 0
    retryable: bool = False

    async def publish(self, event):
        # Once any delta is queued, this logical request can only replay it;
        # it must never start another backend attempt.
        if any(choice.get("delta") or choice.get("text") for choice in event.get("choices", [])):
            self.saw_delta = True
        async with self.changed:
            self.events.append(event)
            self.changed.notify_all()


class PythonFrontend:
    def __init__(self, settings):
        self.settings, self.jobs = settings, OrderedDict()
        self.loop_guard = LoopGuard()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(connect=settings.connect, read=None, write=30, pool=settings.connect))

    def request(self, request_id, body, chat, session_id):
        digest = hashlib.sha256(json.dumps([body, chat, session_id], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        now = time.monotonic()
        for key, job in list(self.jobs.items()):
            if job.done and not job.subscribers and (now - job.finished > self.settings.ttl or len(self.jobs) >= self.settings.max_entries):
                del self.jobs[key]
        job = self.jobs.get(request_id)
        if job:
            if job.digest != digest:
                raise ApiError("X-Request-ID already belongs to a different request body or session", "idempotency_conflict")
            if not (job.done and job.retryable and not job.saw_delta):
                return job
        job = Job(digest, request_id, body, chat, session_id)
        self.jobs[request_id] = job
        job.task = asyncio.create_task(self.run(job))
        return job

    async def run(self, job):
        try:
            await asyncio.wait_for(self.backend(job), self.settings.total)
        except asyncio.CancelledError:
            job.error = ApiError("request cancelled", "request_cancelled")
        except (asyncio.TimeoutError, TimeoutError):
            job.error = ApiError("backend total or idle timeout", "backend_timeout")
        except ApiError as exc:
            job.error = exc
        except ToolProtocolError as exc:
            job.error = ApiError.protocol(exc, tool=bool(job.chat and job.body.get("tools")))
        except ImageError as exc:
            job.error = ApiError(str(exc), "image_invalid")
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            job.error = ApiError("invalid or interrupted backend response: " + str(exc), "upstream_disconnect")
        except Exception as exc:
            job.error = ApiError("frontend request failed: " + str(exc), "internal_error")
        finally:
            if job.error:
                job.error.request_id = job.request_id
            async with job.changed:
                # Cache the digest/result, not the request's RGB payloads.
                job.body = {}
                job.done, job.finished = True, time.monotonic()
                job.changed.notify_all()

    async def backend(self, job):
        # Process images once across protocol attempts and same-ID subscribers.
        body = truncate_tool_messages(job.body) if job.chat else job.body
        body = await prepare_images(body, job.chat)
        if job.chat:
            decision = self.loop_guard.check(job.session_id or job.request_id, body.get("messages", []))
            if not decision.allow:
                raise ApiError(decision.message, decision.code)
        functions = tools_for(body) if job.chat else {}
        constraint = tool_constraint_build(body, functions) if job.chat else None
        created, model = int(time.time()), body.get("model", "souris")
        pending = continuation = None
        retry_prompt = None
        current_attempt = 0
        reasoning = None
        think = job.chat and thinking(body)
        # Clients without affinity still get a valid, request-local transaction.
        tool_session = job.session_id or "gateway-" + job.request_id

        async def transaction(action, payload, retry=False):
            for attempt in range(3):
                try:
                    response = await self.client.post(self.settings.backend + "/v1/session/" + action,
                        json=payload, headers={"X-Retry-Continue": "1"} if retry else {}, timeout=max(60, self.settings.idle))
                    if response.status_code == 200:
                        return True
                    if response.status_code < 500:
                        return False
                except httpx.HTTPError:
                    pass
                if attempt < 2:
                    await asyncio.sleep(0.05 * (attempt + 1))
            return False

        async def settle(commit, retry=False):
            nonlocal pending, continuation
            payload = pending or continuation
            if payload is None:
                return
            publish = commit and current_attempt == 0
            action = "commit" if publish else "abort"
            try:
                if await transaction(action, payload, retry):
                    pending = None
                    continuation = payload if retry else None
                    return
                # A failed finalize must never publish tools or strand admission.
                await transaction("abort", payload)
                pending = continuation = None
                raise ApiError("could not finalize deferred session; no assistant turn was published", "commit_failed")
            except asyncio.CancelledError:
                await asyncio.shield(transaction("abort", payload))
                pending = continuation = None
                raise

        def chunk(delta, finish=None):
            choice = {"index": 0, "delta": delta, "finish_reason": finish} if job.chat else {"index": 0, "text": delta.get("content", ""), "logprobs": None, "finish_reason": finish}
            return {"id": job.request_id, "object": "chat.completion.chunk" if job.chat else "text_completion", "created": created, "model": model, "choices": [choice]}

        async def generate(attempt_body, protocol_attempt=0):
            nonlocal model, pending, continuation, reasoning, retry_prompt, current_attempt
            current_attempt = protocol_attempt
            if protocol_attempt:
                if continuation is None or retry_prompt is None:
                    raise ApiError("protocol retry has no retained prefix lease", "invalid_request")
                # Preserve every original byte, including the official generation header.
                # Only an internal system suffix and the next official header are appended.
                header = "<think>\n" if think else "<think>\n\n</think>\n\n"
                suffix = ("\n</think>" if think else "") + "<|im_end|>\n<|im_start|>system\n"
                prompt = retry_prompt + suffix + attempt_body["_retry_directive"] + "<|im_end|>\n<|im_start|>assistant\n" + header
            else:
                prompt = chat_template(attempt_body, functions) if job.chat else body["prompt"]
            retry_prompt = prompt
            raw = {key: body[key] for key in ("model", "max_tokens", "temperature", "top_p", "top_k", "min_p", "stop", "seed") if key in body}
            raw.update(prompt=prompt, stream=True)
            if constraint:
                raw["constraint"] = constraint
                if think:
                    # Reasoning is ordinary TEXT; enforce required/named after its close.
                    raw["constraint"] = {**constraint, "choice": "auto"}
                    if isinstance(body.get("tool_choice"), dict):
                        name = body["tool_choice"]["function"]["name"]
                        raw["constraint"]["tools"] = [fn for fn in constraint["tools"] if fn["name"] == name]
            if body.get("_images"):
                raw["images"] = body["_images"]
            headers = {"X-Request-ID": job.request_id}
            if functions:
                headers.update({"X-Session-ID": tool_session, "X-Commit-Mode": "deferred"})
                if continuation:
                    headers.update({"X-Retry-Token": continuation["commit_token"],
                                    "X-Retry-Request-ID": continuation["request_id"]})
            elif not functions:
                headers["X-Session-ID"] = job.session_id
            for transport_attempt in range(3):
                if functions:
                    headers["X-Request-ID"] = job.request_id + "-gw-" + uuid.uuid4().hex
                received, reason, usage, parts = False, "stop", {}, []
                parser = _ReasoningStream() if think else None
                try:
                    async with self.client.stream("POST", self.settings.backend + "/v1/completions", json=raw, headers=headers) as response:
                        if response.status_code >= 400:
                            data = await response.aread()
                            try:
                                payload = json.loads(data)
                            except ValueError:
                                payload = {}
                            failure = ApiError.backend(payload, response.status_code, request_id=job.request_id)
                            if functions and not job.saw_delta and failure.code in ("tool_truncated", "tool_protocol_error"):
                                raise ToolProtocolError("truncated" if failure.code == "tool_truncated" else "invalid_response")
                            unaccepted = response.headers.get("X-Souris-Accepted") == "false"
                            if (response.status_code >= 500 and failure.retryable and transport_attempt < 2 and
                                    unaccepted and not job.saw_delta):
                                await asyncio.sleep(0.1 * (transport_attempt + 1))
                                continue
                            job.retryable = failure.retryable and not job.saw_delta and (unaccepted or failure.code == "queue_wait_timeout")
                            raise failure
                        if functions:
                            token = response.headers.get("X-Commit-Token")
                            if not token:
                                raise ApiError("backend omitted deferred commit token", "commit_failed")
                            pending = {"session_id": tool_session, "request_id": headers["X-Request-ID"], "commit_token": token}
                            continuation = None
                        iterator = response.aiter_lines().__aiter__()
                        received_done = False
                        while True:
                            try:
                                line = await asyncio.wait_for(anext(iterator), self.settings.idle if received else None)
                            except StopAsyncIteration:
                                break
                            if received_done or not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                # Read the HTTP end, not only the SSE terminal marker.
                                received_done = True
                                continue
                            value = json.loads(data)
                            if "error" in value:
                                detail = value["error"]
                                code = detail.get("code") if isinstance(detail, dict) else None
                                if functions and not job.saw_delta:
                                    if code in ("tool_truncated", "tool_protocol_error"):
                                        raise ToolProtocolError("truncated" if code == "tool_truncated" else "invalid_response")
                                    if not code and parts:
                                        # Legacy peers have no machine code. Classify the actual
                                        # model turn with the same assembler, never the error text.
                                        partial = parser.finish()[1] if parser else "".join(parts)
                                        assembler = ToolCallAssembler(functions, body, job.request_id)
                                        assembler.push(partial)
                                        assembler.finish(reason)
                                raise ApiError.backend(value, 500, phase="decode", request_id=job.request_id)
                            model = value.get("model", model)
                            choice = value["choices"][0]
                            part = choice.get("text", "")
                            if part:
                                if not functions:
                                    if not received and job.chat:
                                        await job.publish(chunk({"role": "assistant"}))
                                    for key, fragment in (parser.feed(part) if parser else [("content", part)]):
                                        await job.publish(chunk({key: fragment}))
                                elif parser:
                                    parser.feed(part)
                                parts.append(part)
                                received = True
                            if choice.get("finish_reason"):
                                reason = choice["finish_reason"]
                            usage = value.get("usage", usage)
                        if not received_done:
                            raise ApiError("backend stream ended without completion", "upstream_disconnect")
                        if parser:
                            for key, fragment in parser.feed("", final=True):
                                if not functions:
                                    await job.publish(chunk({key: fragment}))
                            reasoning, text = parser.finish()
                        else:
                            text = "".join(parts)
                        return text, reason, usage, model
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if received or job.saw_delta or transport_attempt == 2:
                        job.retryable = not received
                        raise
                    await asyncio.sleep(0.1 * (transport_attempt + 1))

        calls = []
        if functions:
            retry = GatewayRetry(functions, body, job.request_id)
            try:
                text, calls, reason, usage, model = await retry.runModelTurn(generate, settle)
            finally:
                if pending or continuation:
                    await asyncio.shield(settle(False))
            # Publish only the validated whole turn, including ordinary prose.
            await job.publish(chunk({"role": "assistant"}))
            if think and reasoning:
                await job.publish(chunk({"reasoning_content": reasoning}))
            if text:
                await job.publish(chunk({"content": text}))
            if calls:
                await job.publish(chunk({"tool_calls": [{"index": i, **call} for i, call in enumerate(calls)]}))
        else:
            text, reason, usage, model = await generate(body)
            if job.chat and not job.saw_delta:
                await job.publish(chunk({"role": "assistant"}))
        if job.chat:
            message = {"role": "assistant", "content": text or None}
            if think:
                message["reasoning_content"] = reasoning
            if calls:
                message["tool_calls"] = calls
                reason = "tool_calls"
            choices = [{"index": 0, "message": message, "finish_reason": reason}]
        else:
            choices = [{"index": 0, "text": text, "logprobs": None, "finish_reason": reason}]
        await job.publish(chunk({}, reason))
        if body.get("stream_options", {}).get("include_usage"):
            await job.publish({"id": job.request_id, "object": "chat.completion.chunk" if job.chat else "text_completion", "created": created, "model": model, "choices": [], "usage": usage})
        job.response = {"id": job.request_id, "object": "chat.completion" if job.chat else "text_completion", "created": created, "model": model, "choices": choices, "usage": usage}

    async def wait(self, job, request, first):
        began = time.monotonic()
        while not job.done and not (first and job.events):
            if await request.is_disconnected():
                raise ApiError("client disconnected", "client_disconnected")
            if first and time.monotonic() - began >= self.settings.ttft:
                raise ApiError("TTFT timeout; retry with the same X-Request-ID to wait for the existing task", "ttft_timeout")
            async with job.changed:
                try:
                    await asyncio.wait_for(job.changed.wait(), 0.1)
                except asyncio.TimeoutError:
                    pass
        if job.error and (not first or not job.events):
            raise job.error

    async def detach(self, job, cancel):
        job.subscribers -= 1
        if cancel and not job.subscribers and not job.done:
            job.task.cancel()
            # Cancellation can precede the task's first coroutine instruction.
            # Publish the terminal state here as well as in run's finally.
            async with job.changed:
                job.error = ApiError("request cancelled", "request_cancelled")
                job.done, job.finished = True, time.monotonic()
                job.changed.notify_all()


class _CompletionStream(StreamingResponse):
    """Own the subscription even if disconnect precedes iterator startup."""
    def __init__(self, content, frontend, job, **kwargs):
        super().__init__(content, **kwargs)
        self.frontend, self.job = frontend, job

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await asyncio.shield(self.frontend.detach(self.job, cancel=not self.job.done))


def create_app(settings=None):
    settings = settings or Settings()
    @asynccontextmanager
    async def lifespan(app):
        frontend = PythonFrontend(settings)
        app.state.frontend = frontend
        yield
        for job in frontend.jobs.values():
            if not job.done:
                job.task.cancel()
        await asyncio.gather(*(j.task for j in frontend.jobs.values()), return_exceptions=True)
        await frontend.client.aclose()
    app = FastAPI(lifespan=lifespan)
    @app.exception_handler(ApiError)
    async def invalid(request, error):
        error.request_id = error.request_id or request.headers.get("X-Request-ID") or uuid.uuid4().hex
        return render_http(error)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return await invalid(request, ApiError.backend({}, error.status_code, phase="admission"))

    @app.exception_handler(Exception)
    async def unexpected(request, error):
        return await invalid(request, ApiError("frontend request failed", "internal_error"))

    async def proxy(request, path):
        try:
            response = await request.app.state.frontend.client.get(settings.backend + path)
        except httpx.HTTPError as exc:
            raise ApiError("backend unavailable", "engine_unavailable") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiError("invalid backend response", "upstream_disconnect") from exc
        if response.status_code >= 400:
            raise ApiError.backend(payload, response.status_code, phase="admission")
        return JSONResponse(payload)

    @app.get("/health")
    async def health(request: Request):
        return await proxy(request, "/health")

    @app.get("/v1/models")
    async def models(request: Request):
        return await proxy(request, "/v1/models")

    async def complete(request, chat):
        try:
            body = await request.json()
        except ValueError as exc:
            raise ApiError("invalid JSON") from exc
        validate(body, chat)
        if chat:
            chat_template(truncate_tool_messages(body), tools_for(body))
        request_id = request.headers.get("X-Request-ID") or request.headers.get("Idempotency-Key") or "cmpl-" + uuid.uuid4().hex
        session_id = (request.headers.get("X-Session-ID") or request.headers.get("x-session-affinity") or
                      request.headers.get("session_id", ""))
        frontend = request.app.state.frontend
        job = frontend.request(request_id, body, chat, session_id)
        job.subscribers += 1
        try:
            await frontend.wait(job, request, True)
            if not body.get("stream"):
                await frontend.wait(job, request, False)
        except ApiError as exc:
            # A TTFT wait timeout is a completed HTTP error response, not a
            # client disconnect. Preserve the task for same-ID retries.
            await frontend.detach(job, cancel=exc.code != "ttft_timeout")
            exc.request_id = request_id
            raise
        except asyncio.CancelledError:
            await frontend.detach(job, cancel=True)
            raise
        if not body.get("stream"):
            await frontend.detach(job, cancel=False)
            return JSONResponse(job.response, headers={"X-Request-ID": request_id})
        async def events():
            try:
                cursor = 0
                while True:
                    while cursor < len(job.events):
                        yield "data: " + json.dumps(job.events[cursor], ensure_ascii=False) + "\n\n"
                        cursor += 1
                    if job.done:
                        if job.error:
                            yield render_sse(job.error)
                        yield "data: [DONE]\n\n"
                        break
                    async with job.changed:
                        if cursor == len(job.events) and not job.done:
                            await job.changed.wait()
            except asyncio.CancelledError:
                raise
            except Exception:
                yield render_sse(ApiError("frontend stream failed", "internal_error",
                                          phase="stream", request_id=request_id))
                yield "data: [DONE]\n\n"
        return _CompletionStream(events(), frontend, job, media_type="text/event-stream", headers={"X-Request-ID": request_id, "Cache-Control": "no-cache"})
    @app.post("/v1/completions")
    async def raw(request: Request):
        return await complete(request, False)
    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await complete(request, True)
    return app


if __name__ == "__main__":
    import uvicorn
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", default=Settings.backend)
    cli.add_argument("--host", default="127.0.0.1")
    cli.add_argument("--port", type=int, default=8080)
    for field_name in ("connect", "ttft", "idle", "total", "ttl"):
        cli.add_argument("--" + field_name, type=float, default=getattr(Settings, field_name))
    args = cli.parse_args()
    values = {name: getattr(args, name) for name in ("backend", "connect", "ttft", "idle", "total", "ttl")}
    if any(values[name] <= 0 for name in ("connect", "ttft", "idle", "total", "ttl")):
        cli.error("timeouts and TTL must be positive")
    uvicorn.run(create_app(Settings(**values)), host=args.host, port=args.port)
