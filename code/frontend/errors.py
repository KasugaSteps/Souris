"""Public frontend errors: one code table, one HTTP/SSE representation."""
from dataclasses import dataclass
import json

from fastapi.responses import JSONResponse


@dataclass(frozen=True)
class ErrorRule:
    status: int
    type: str
    phase: str
    retryable: bool = False
    retry_after: str | None = None


ERROR_RULES = {
    "invalid_request": ErrorRule(400, "invalid_request_error", "admission"),
    "context_overflow": ErrorRule(400, "invalid_request_error", "admission"),
    "image_invalid": ErrorRule(400, "invalid_request_error", "admission"),
    "auth_failed": ErrorRule(401, "invalid_request_error", "admission"),
    "forbidden": ErrorRule(403, "invalid_request_error", "admission"),
    "not_found": ErrorRule(404, "invalid_request_error", "admission"),
    "method_not_allowed": ErrorRule(405, "invalid_request_error", "admission"),
    "model_busy": ErrorRule(409, "model_busy", "admission"),
    "commit_conflict": ErrorRule(409, "conflict_error", "commit"),
    "idempotency_conflict": ErrorRule(409, "conflict_error", "admission"),
    "tool_truncated": ErrorRule(422, "tool_error", "tool_parse"),
    "tool_protocol_error": ErrorRule(422, "tool_error", "tool_parse"),
    "no_progress": ErrorRule(422, "tool_error", "agent_loop"),
    "poll_timeout": ErrorRule(422, "tool_error", "agent_loop"),
    "queue_wait_timeout": ErrorRule(429, "rate_limit_error", "admission", True, "1"),
    "request_cancelled": ErrorRule(499, "server_error", "decode"),
    "client_disconnected": ErrorRule(499, "server_error", "stream"),
    "internal_error": ErrorRule(500, "server_error", "decode"),
    "commit_failed": ErrorRule(500, "server_error", "commit"),
    "structured_output_failed": ErrorRule(502, "server_error", "decode"),
    "upstream_disconnect": ErrorRule(502, "server_error", "decode", True),
    "engine_starting": ErrorRule(503, "server_error", "admission", True),
    "engine_unavailable": ErrorRule(503, "server_error", "admission", True),
    "ttft_timeout": ErrorRule(504, "server_error", "prefill", True),
    "backend_timeout": ErrorRule(504, "server_error", "decode"),
}
# Legacy peers without codes retain generic HTTP meaning, never message heuristics.
# In particular legacy 503 cannot be inferred to be queue timeout or model busy.
LEGACY_CODES = {ERROR_RULES[code].status: code for code in (
    "invalid_request", "auth_failed", "forbidden", "not_found", "method_not_allowed",
    "commit_conflict", "tool_protocol_error", "queue_wait_timeout", "request_cancelled",
    "internal_error", "upstream_disconnect", "engine_unavailable", "backend_timeout",
)}


class ApiError(Exception):
    def __init__(self, message, code="invalid_request", *, phase=None, request_id=None):
        super().__init__(message)
        self.code = code
        self.rule = ERROR_RULES[code]
        self.message = message
        self.phase = phase or self.rule.phase
        self.request_id = request_id

    @property
    def status(self):
        return self.rule.status

    @property
    def retryable(self):
        return self.rule.retryable and self.phase != "stream"

    def body(self, *, stream=False):
        return {"error": {
            "message": self.message, "type": self.rule.type, "param": None,
            "code": self.code, "phase": "stream" if stream else self.phase,
            "retryable": False if stream else self.retryable,
            "request_id": self.request_id,
        }}

    @classmethod
    def backend(cls, payload, status, *, phase=None, request_id=None):
        detail = payload.get("error", {}) if isinstance(payload, dict) else {}
        detail = detail if isinstance(detail, dict) else {}
        code = detail.get("code") or LEGACY_CODES.get(status, "internal_error")
        if not isinstance(code, str) or code not in ERROR_RULES:
            code = "internal_error"
        message = detail.get("message")
        return cls(message if isinstance(message, str) else code.replace("_", " "), code,
                   phase=phase, request_id=request_id)

    @classmethod
    def protocol(cls, error, *, tool=False):
        code = ("tool_truncated" if error.code == "truncated" else
                "structured_output_failed" if error.code == "incomplete_reasoning" and not tool else
                "tool_protocol_error")
        return cls(str(error), code)


def render_http(error):
    headers = {"X-Request-ID": error.request_id} if error.request_id else {}
    if error.rule.retry_after:
        headers["Retry-After"] = error.rule.retry_after
    return JSONResponse(error.body(), status_code=error.status, headers=headers)


def render_sse(error):
    return "event: error\ndata: " + json.dumps(error.body(stream=True), ensure_ascii=False) + "\n\n"
