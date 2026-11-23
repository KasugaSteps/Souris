"""Atomic tool-turn validation and bounded gateway-only protocol recovery."""
import asyncio
import hashlib
import json
import re

import jsonschema


class ToolProtocolError(Exception):
    def __init__(self, code, text=""):
        super().__init__("invalid tool response: " + code)
        self.code = code
        self.retryable = True
        self.raw_snippet = text[:160]


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("non-finite JSON constant")


class ToolCallAssembler:
    """Expose a turn only after every canonical call passes validation."""
    def __init__(self, functions, body, request_id):
        self.functions, self.body, self.request_id = functions, body, request_id
        self.reset()

    def reset(self):
        self.parts = []

    def push(self, delta):
        self.parts.append(delta)

    def finish(self, reason="stop"):
        raw = "".join(self.parts)
        choice = self.body.get("tool_choice", "auto")
        if reason == "length" and (choice == "required" or isinstance(choice, dict) or "<tool_call" in raw):
            raise ToolProtocolError("truncated", raw)
        decoder = json.JSONDecoder(object_pairs_hook=_object, parse_constant=_constant)
        text, values, position = [], [], 0
        while position < len(raw):
            start = raw.find("<tool_call>", position)
            outside = raw[position:start if start >= 0 else len(raw)]
            if re.search(r"</?\s*(?:function|invoke|parameter)\b|</tool_call>|<tool_call\b", outside):
                raise ToolProtocolError("legacy_format", outside)
            for match in re.finditer(r'\{', outside):
                try:
                    bare, _ = decoder.raw_decode(outside, match.start())
                except ValueError:
                    fragment = outside[match.start():]
                    if re.match(r'\{\s*"(?:name|function|arguments|parameters)"\s*:', fragment) and re.search(r'"(?:arguments|parameters)"\s*:', fragment):
                        raise ToolProtocolError("invalid_json", outside)
                    continue
                if isinstance(bare, dict) and (set(bare) >= {"name", "arguments"} or set(bare) >= {"function", "parameters"}):
                    raise ToolProtocolError("invalid_json", outside)
            text.append(outside)
            if start < 0:
                break
            position = start + len("<tool_call>")
            position += len(raw[position:]) - len(raw[position:].lstrip())
            if re.match(r"</?\s*(?:function|invoke|parameter)\b", raw[position:]):
                raise ToolProtocolError("legacy_format", raw[position:])
            try:
                value, end = decoder.raw_decode(raw, position)
            except ValueError as exc:
                raise ToolProtocolError("invalid_json", raw[position:]) from exc
            end += len(raw[end:]) - len(raw[end:].lstrip())
            if not raw.startswith("</tool_call>", end):
                raise ToolProtocolError("truncated", raw[end:])
            position = end + len("</tool_call>")
            if not isinstance(value, dict) or set(value) != {"name", "arguments"} or not isinstance(value["arguments"], dict):
                raise ToolProtocolError("invalid_json")
            name = value["name"]
            if not isinstance(name, str) or name not in self.functions:
                raise ToolProtocolError("unknown_tool")
            choice = self.body.get("tool_choice", "auto")
            if choice == "none" or isinstance(choice, dict) and name != choice["function"]["name"]:
                raise ToolProtocolError("unknown_tool")
            try:
                jsonschema.Draft202012Validator(self.functions[name]["parameters"]).validate(value["arguments"])
            except jsonschema.ValidationError as exc:
                raise ToolProtocolError("schema_invalid") from exc
            values.append(value)
            limit = self.body.get("max_calls", 4) if self.body.get("parallel_tool_calls", True) else 1
            if len(values) > limit:
                raise ToolProtocolError("too_many_calls")
        content = "".join(text)
        choice = self.body.get("tool_choice", "auto")
        if choice == "required" or isinstance(choice, dict):
            if content.strip() or not values:
                raise ToolProtocolError("invalid_json", content)
        prefix = hashlib.sha256(self.request_id.encode()).hexdigest()[:16]
        calls = [{"id": f"call_{prefix}_{index}", "type": "function", "function": {
            "name": value["name"], "arguments": json.dumps(value["arguments"], ensure_ascii=False)}}
            for index, value in enumerate(values)]
        return content, calls


_RETRY_HINTS = {
    "truncated": "The previous tool call was cut off inside its arguments at max_tokens. "
                 "Reason briefly about the smallest sufficient change, then emit one complete call. "
                 "Split write/edit into at most 40 lines or 1KB per call.",
    "invalid_json": "The previous response contained invalid JSON inside a tool call. "
                    "Locate the malformed block, fix it, and return one complete canonical call.",
    "legacy_format": "The previous response used legacy XML at the tool boundary. "
                     "Replace it with one canonical <tool_call> JSON block.",
    "schema_invalid": "The previous arguments did not match the declared schema. "
                      "Fix the field names, types, and required values, then return one complete call.",
    "unknown_tool": "The previous tool name was not declared. "
                    "Choose only from the supplied tool names.",
    "too_many_calls": "The previous response exceeded the allowed number of tool calls. "
                      "Return no more calls than max_calls allows.",
    "invalid_response": "The previous response produced no complete assistant/tool turn. "
                        "Return one complete canonical tool call.",
}


class GatewayRetry:
    """Retry invalid model protocol, never provider failures or executed tools."""
    maxProtocolRetries = 2

    def __init__(self, functions, body, request_id):
        self.functions, self.body, self.request_id = functions, body, request_id
        self.failures = []

    def discardFailedTurn(self, error):
        # Retain each distinct code once: failed text and arguments never become history.
        if error.code not in self.failures:
            self.failures.append(error.code)

    def retryPolicy(self, error, attempt):
        return error.retryable and attempt < self.maxProtocolRetries and self.body.get("tool_choice") != "none"

    def buildInternalRetryDirective(self):
        # Every retry carries all earlier failures, not just the latest one.
        hints = []
        for code in self.failures:
            hint = _RETRY_HINTS.get(code, "The previous response failed at tool parsing. "
                                         "Reason briefly about the failure, then correct that exact point.")
            if hint not in hints:
                hints.append(hint)
        if not hints:
            hints.append("The previous response failed at tool parsing.")
        return ("Previous attempts failed at:\n- " + "\n- ".join(hints) +
                "\nCorrect those points before acting. Return only complete canonical "
                '<tool_call>{"name":"...","arguments":{...}}</tool_call> blocks. '
                "No prose, legacy XML, function/invoke/parameter tags, or bare JSON. "
                "Use only the supplied tools and schema.")

    async def runModelTurn(self, generate, settle=None):
        for attempt in range(self.maxProtocolRetries + 1):
            body = self.body
            if attempt:
                body = {**body, "_retry_directive": self.buildInternalRetryDirective()}
            try:
                raw, reason, usage, model = await generate(body, attempt)
                assembler = ToolCallAssembler(self.functions, self.body, self.request_id)
                assembler.push(raw)
                text, calls = assembler.finish(reason)
                if settle:
                    await settle(True)
                return text, calls, reason, usage, model
            except ToolProtocolError as error:
                again = self.retryPolicy(error, attempt)
                if settle:
                    await settle(False, again)
                self.discardFailedTurn(error)
                if not again:
                    raise
            except BaseException:
                if settle:
                    await asyncio.shield(settle(False))
                raise

def truncate_tool_text(text):
    """Bound model-visible tool output, including the truncation notice."""
    encoded = text.encode("utf-8")
    if len(encoded) <= 4096 and len(text.splitlines()) <= 100:
        return text
    marker = "\n[tool output truncated: head and tail retained]\n"
    budget = (4096 - len(marker.encode())) // 2
    head = encoded[:budget].decode("utf-8", errors="ignore").splitlines()[:49]
    tail = encoded[-budget:].decode("utf-8", errors="ignore").splitlines()[-49:]
    return "\n".join(head) + marker + "\n".join(tail)


def truncate_tool_messages(body):
    messages = []
    for original in body.get("messages", []):
        message = dict(original)
        if message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = truncate_tool_text(content)
            elif isinstance(content, list):
                # Keep images in place; apply one text budget across all text parts.
                joined = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
                shortened = truncate_tool_text(joined)
                if shortened != joined:
                    first = True
                    parts = []
                    for part in content:
                        if part.get("type") != "text":
                            parts.append(part)
                        elif first:
                            parts.append({"type": "text", "text": shortened}); first = False
                    message["content"] = parts
        messages.append(message)
    return {**body, "messages": messages}
