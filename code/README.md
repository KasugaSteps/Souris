# Souris v0.1

The implementation contract is [spec/v0.1.md](spec/v0.1.md). GGML supplies tensor allocation, quantization, MMQ/MMVQ and dense model operations. The validated target is Qwen3.8-Flash-Next with IQ2_XXS gate/up and Q4_0 down projections.

Build on Linux with CUDA and OpenSSL development files (`libssl-dev` on Ubuntu), then select the raw CLI or raw completion server through one startup TOML file.

```sh
cmake -S . -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda --target souris -j8
cp souris.toml.example souris.toml
build-cuda/bin/souris chat --config souris.toml
build-cuda/bin/souris serve --config souris.toml
```

`chat` accepts raw text and retains its conversation. `/reset` clears the session and `/exit` exits. The C++ server provides `/health`, `/v1/models` and `/v1/completions`. OpenAI chat/tool support and pi-mono configuration are in [frontend/README.md](frontend/README.md).

`max_prompt_tokens` defaults to `context`. Both commands require `0 < max_prompt_tokens <= context` and lock the budget at startup. Oversized requests return 400. `context` bounds prompt plus completion tokens. `staging_mib`, `read_threads`, `mmq_group`, optional GPU/RAM byte budgets, MTP assets, server port and logging are configured in TOML.

- `Runtime` owns Device, Model, Engine and SessionManager in lifetime order. Model owns the SRS1 repository, its embedded tokenizer, and an optional VisionEncoder. `Engine(Model&, Device&, Config&)` borrows these resources and owns KV and startup planning. It measures free VRAM after permanent allocations, reserves 1 GiB, and derives `D_total` from `Layout.record_bytes`. Prefill borrows hidden/workspace tail slots and returns them before Decode.
- `ExpertRepository` maps dense shard ranges and reads expert records through aligned `O_DIRECT`. `Layout` derives role geometry and the ordered quantization signature from the SRS1 v1 header. SlotPool, ExpertCache, StagingRing, TransferEngine and MoE use that same layout.
- `ExpertCache` holds pageable records. Server auto capacity is bounded by the complete expert region and available RAM minus 8 GiB. Startup warmup uses 128 Prefill tokens and one Decode token, resets execution/cache state, then blocking preload fills complete RAM layers.
- `SlotPool` owns final GPU expert slots, the pointer table, SLRU and MTP provisional transactions. `TransferEngine` reuses pinned record staging and publishes copied events; compute waits on the current MMQ group's uploads.
- `Scheduler` runs layer-major Prefill with 1024-token tiles. Route grouping stays on GPU. Host receives counts and schedules the unique layer union in physical record order. `MoEKernel` uses one gate/up MMQ Product, fused SwiGLU/Q8_1 epilogue, one down Product and weighted scatter. Default group size is 16; the prepared workspace permits groups up to 64.
- `NativeLayerKernels` supplies Decode and MTP verify pre/post graphs. Persistent KV is Q8_0, external Decode batch is one, and typed/count captures are built at startup. Optional `DraftRunner` uses its resident Q4_0 assets and private KV; `VerifyEngine` commits accepted target state and provisional experts in place.
- `SessionManager` retains the last session and at most eight GDN/PLE checkpoints. Prefix reuse restores state at length minus one and recomputes the boundary token. K/V and index remain in their original buffers. MTP prompt KV is rebuilt from the recomputed suffix. Successful raw response delivery publishes history; disconnected requests abandon it.
- `OpsLog` writes request JSONL to stdout and an optional daily file. It contains operational metrics and identifiers. Startup budgets are printed once to stderr.

Build a new SRS1 v1 asset directly from source GGUF shards after building GGML.

```sh
python3 tools/srs1.py --plan --output /data/new-souris.bin /path/to/source.gguf
python3 tools/srs1.py --output /data/new-souris.bin /path/to/source.gguf
```

The converter rebuilds GGUF metadata with canonical `general.name = Qwen3.8 Flash Next`, recomputes shard sizes/data offsets and the SRS1 expert region, and preserves source bytes, validates indexer/QSA structure, records authorized embedded metadata normalization, and checks dense/role/header checksums before publishing the container and its manifest. Gate and up require matching type and row geometry; down has its own type. Shared experts remain dense tensors.

Type traits and MMQ dispatch are prepared for Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q8_0, Q4_0, Q4_K, Q5_K, Q5_1 and Q6_K. The tested target combination is `(IQ2_XXS, IQ2_XXS, Q4_0)`; MTP Q4_0 has regression coverage. All other target quantizations remain **untested**.

Kernel comparisons, conversion checks and API acceptance artifacts are under [reports/product-mmq](reports/product-mmq). The older native path's real-code top1 nondeterminism remains [KNOWN-DEFERRED](reports/product-mmq/KNOWN-DEFERRED.md).

Static PNG/JPEG/WebP/BMP input is supported through the Python frontend after configuring `vision.mmproj`. Python applies EXIF, gray-127 alpha composition and a centered 768×768 letterbox. C++ accepts RGB8 and encodes exactly 576 image rows per picture through the paired Qwen3VL mmproj; image rows replace only their `<|image_pad|>` spans before HC repetition. Up to four pictures are allowed. Image hashes and token spans participate in prefix matching; a changed image invalidates reuse. OpsLog keeps the same request fields.
