# Souris OpenAI frontend

The C++ process accepts raw text. This Python adapter renders Qwen text/tool messages and exposes OpenAI chat/completions to pi-mono clients.

```sh
cp souris.toml.example souris.toml
build-cuda/bin/souris serve --config souris.toml
python3 -m pip install virtualenv
python3 -m virtualenv frontend/.venv
frontend/.venv/bin/pip install -r frontend/requirements.txt
frontend/.venv/bin/python frontend/server.py --backend http://127.0.0.1:8000 --port 8080
```

Python 3.10 or later is required. The dependency versions are pinned to the validated cloud environment.

Use `http://127.0.0.1:8080/v1` as the OpenAI-compatible base URL. The model ID comes from `/v1/models`. Configure the pi-mono OpenAI-completions provider to send `max_tokens`, text content, and function tools. Streaming uses indexed `tool_calls` deltas and ends with `[DONE]`.

`pi-models.example.json` contains provider settings for pi-mono. Merge its provider into `~/.pi/agent/models.json` and adjust the URL/context to your deployment. Its session-affinity option forwards pi's session identity into the backend's `X-Session-ID`.

`souris chat --config souris.toml` runs the raw interactive CLI. `/reset` clears its session and `/exit` exits. It applies no chat template.

`max_prompt_tokens` defaults to `context`. Both commands validate `0 < max_prompt_tokens <= context` before allocating/capturing. Requests exceeding the locked prompt limit or `prompt_tokens + max_tokens > context` receive HTTP 400.

The frontend supports system/user/assistant/tool text, function schemas and results, tool_choice, parallel_tool_calls, temperature/top_p/seed, stop strings, streamed usage, and prompt-hinted JSON-object responses. It supports up to four static PNG/JPEG/WebP/BMP images. It explicitly rejects animated images, audio/video, logprobs, n>1, and constrained JSON schemas.

Supply `X-Request-ID` to replay a completed result or join the same in-flight request. Reusing an ID with a different body, endpoint, or session returns 409. Results and cancellation tombstones remain for a bounded TTL (default 600 seconds, at most 256 completed entries). This registry is process-local. Run one frontend worker for one backend.

`Idempotency-Key` is accepted when `X-Request-ID` is absent. Callers retrying a timed-out request must reuse one of these identifiers. Requests without an identifier receive a new ID in the response headers.

`X-Session-ID` identifies the last active session. Same-session prefixes restore GPU GDN/PLE checkpoints and recompute from their final token. Caches and captured Decode graphs survive requests. A session change clears prefix state.

The frontend also maps pi's `x-session-affinity` / `session_id` headers to this session identity. Explicit `X-Session-ID` takes priority.

`--connect`, `--ttft`, `--idle`, and `--total` set timeout seconds. A TTFT timeout returns 504 while retaining the original task. Retry the same ID to wait for it. Transport retries never resend an accepted request. Tool protocol recovery below continues the same admitted logical request after discarding an invalid whole turn. Connection establishment failures and explicitly unaccepted 5xx responses can retry before any delta. Actual client disconnection closes the backend response and cancels unfinished generation. C++ commits session history only after successful raw response delivery.

Tools accept only canonical `<tool_call>{"name":"...","arguments":{...}}</tool_call>` blocks. `ToolCallAssembler` rejects XML/function/invoke/parameter wrappers, aliases, unknown names, invalid JSON/schema and excess calls. Every call must pass before any assistant text or tool call is delivered. Tool execution belongs to Pi/the client; the gateway never executes tools.

Tool requests now send a structured `constraint` to the raw backend. `required` and named choices constrain the first output token; `auto` scans only the reserved `<tool_call>` marker in text, then constrains the JSON body and closing tag. Text does not request a vocabulary mask. Internal token-trie/state caches keep masks on the GPU; temperature and top_p still use the existing sampler.

`max_calls` defaults to 4 and accepts 1..8. `parallel_tool_calls=false` forces one call. Each returned call has a stable unique ID; every call needs exactly one result before another generation, including an explicit failure result when execution failed. The frontend does not execute or retry tools. `GatewayRetry` may regenerate an invalid assistant turn at most twice, using only a temporary internal system directive. It discards all failed text/calls, keeps the original user messages and official generation header, and never adds an assistant prefill or a user repair prompt. Exhaustion returns a non-retryable 422 tool_truncated/tool_protocol_error. Provider/context errors do not trigger protocol recovery. Failed attempts expose no calls to execute; replaying the same completed request ID returns the same validated call IDs without another backend request.

The grammar subset covers object/array/string/number/integer/boolean/null, properties, required, additionalProperties=false (or unconstrained extra keys), items, enum and const. Description/default are ignored. Unsupported schema keywords, unsatisfiable constants and uncompileable schemas return 400. Compilation currently limits objects to 12 declared printable-ASCII property names and nesting to 24; escaped/non-ASCII property names are rejected. String values support UTF-8 and JSON escapes. This is not complete JSON Schema support.

When MTP is configured, auto TEXT uses it until the opening tool marker. Verification leaves the marker-completing token pending instead of committing speculative body rows. Tool bodies use target Decode; after closing, the drafter restarts from the last real target cell. That restart discards old draft KV, so proposal acceptance may temporarily decrease. Required/named requests use target Decode throughout. Unconstrained requests retain their existing MTP path.

`prefill_ms` ends after target Prefill/head and expert drain. First-token sampling and its grammar mask-cache work are reported separately as `prefill_sample_ms`; schema configure happens before Prefill. Request wall time still includes this work. Request counters expose `tool_mtp_entries` and `tool_body_mtp_drafts` for phase verification.

Operational logs contain one JSON object per request and no prompt/token contents. File logs rotate as `<log.file>.YYYY-MM-DD.jsonl` in UTC.

## Protocol references

- [Qwen tool-call formatting](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)
- [pi-mono provider and streaming tool-call documentation](https://github.com/badlogic/pi-mono/tree/main/packages/ai)
- [toml++ used by the startup configuration parser](https://github.com/marzer/tomlplusplus/tree/v3.4.0)

## Images

Configure `[vision]` and its `mmproj` path on the raw backend. The model's embedded tokenizer and paired vision metadata are checked at load time. With no mmproj, image requests return 400.

Chat content accepts standard OpenAI `image_url` parts (base64 data URIs or public HTTP/HTTPS URLs) and pi-ai `ImageContent` (`type=image`, `data`, `mimeType`). Image order and placement follow the message content. The pi example advertises `input: ["text", "image"]`, including tool-result image forwarding by pi-ai.

Limits are one static frame, at most four images, 10 MB (10,000,000 bytes) per image, 40 MP decoded pixels, edges 16..8192. Detection uses file signatures and frame metadata. GIF, APNG, animated WebP, SVG, video and audio are rejected. EXIF orientation is applied, transparency is composited over RGB(127,127,127), and aspect ratio is preserved by a centered 768×768 letterbox. No crop/stretch or dynamic resolution is used. Public downloads have a 10-second limit; private addresses and unsafe redirects are rejected, and the connection is pinned to the validated DNS result.

Each image becomes 576 `<|image_pad|>` tokens inside vision delimiters and one RGB8 payload. The backend computes the canonical RGB/version hash itself. Same-session changed pixels, order or image positions cannot reuse the old prefix. Same request ID with a different body returns 409. Download/decode is part of that single in-flight task, so retrying a URL request reuses its original pixels/result without fetching again; use a new ID for new content at the same URL. OpsLog adds no image-specific fields.

## Atomic tool turns and session isolation

Requests carrying nonempty `tools` buffer the entire backend assistant turn, even when `stream=true`. After successful whole-batch validation, the gateway emits ordinary OpenAI SSE text/tool-call deltas and the final finish reason. Invalid or truncated batches emit no partial assistant. Text requests without tools keep incremental streaming.

The first tool attempt sends the complete history with `X-Session-ID` and `X-Commit-Mode: deferred`. After atomic validation it commits the staged token through `/v1/session/commit`, then publishes the turn. Invalid output is aborted before retry while retaining one unpublished prompt-end checkpoint and the same RequestGate owner. Attempts 1/2 keep the same session and deferred mode, append only a temporary internal system suffix, and require a proven prefix hit. Successful retries are aborted, never committed; their assistant text and directives do not update committed history. The next normal attempt may need one full Prefill, then prefix reuse resumes. Text-only requests with session affinity and CLI keep immediate commit. Immediate raw requests without a session ID abandon the result instead of creating an anonymous commit. Clients without affinity get a request-local session, so they cannot reuse prefixes across unrelated request IDs.

The raw API defaults to `X-Commit-Mode: immediate`. Deferred requests require a nonempty session and return `X-Commit-Token`; commit/abort RPC bodies contain `session_id`, `request_id` and `commit_token`. A successful stage expires after 30 seconds; generation time does not count. One pending transaction excludes all new completions on this single-session runtime (409). Repeated commits are idempotent; commit-after-abort and abort-after-commit return 409. The frontend retries commit RPCs with the same token and reports 500 if finalization remains uncertain.

Abort restores the protected committed recurrent checkpoint. Only a request that rewrote older KV rows needs to reconstruct the old KV before restoring that checkpoint; the normal append-only path does not replay. Two of the existing eight checkpoint slots are reserved for pending captures; published snapshots are protected throughout the attempt. Rebuilding old KV preserves the original prompt/final-token tile boundary before replaying the generated tail. No extra KV copy is allocated. Terminal commit/abort receipts are retained for the server process lifetime.

Before rendering the template, each tool-result text is capped at 4096 UTF-8 bytes and 100 lines, including a head/tail truncation notice. Image parts remain supported. All call IDs need a result before another model turn; execution failures must also be returned as tool results. Tool requests are validated atomically, not executed concurrently by the gateway.

## Chat reasoning

Chat accepts request-local `reasoning_effort`: `none` (also the default),
`minimal`, `low`, `medium`, or `high`. Enabled efforts insert one
stable system instruction and an open thinking header. They guide the model;
they do not impose a hard reasoning-token budget. The raw completion API is unchanged.

Streaming chat separates `delta.reasoning_content` from `delta.content`;
non-streaming replies use the same two fields in the assistant message. Return
`reasoning_content` unchanged in assistant history. Only template framing is
removed while parsing; the payload is not trimmed. An incomplete thinking block
fails the turn. Tool requests still buffer and validate the entire call batch
before publishing calls or committing the staged turn.

Effort may change between requests in the same session. An unchanged effort
renders the same instruction; a changed effort can miss the existing prefix.
No prefix-matching or session transaction policy is changed.

Internal continuation uses `X-Retry-Continue: 1` on abort, then `X-Retry-Token` and `X-Retry-Request-ID` on the next attempt. The token authenticates the same runtime owner; ordinary completions still queue. Final abort, cancellation and the existing 30-second pending TTL release ownership. These headers are gateway-internal; Pi and CLI are unchanged.
