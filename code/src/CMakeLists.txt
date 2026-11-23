find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
if (NOT GGML_CUDA)
    message(FATAL_ERROR "Souris requires the CUDA backend")
endif ()
add_library(souris-core STATIC
    grammar_verifier.cpp
    runtime.cpp request_gate.cpp image.cpp vision.cpp native_layer.cpp native_ops.cu staging_ring.cpp prefill_ops.cpp prefill_ops.cu repository.cpp expert_cache.cpp transfer.cpp model.cpp tokenizer.cpp device.cpp kv_cache.cpp io.cpp ops.cpp ops.cu moe.cpp moe.cu graph_builder.cpp stats.cpp expert.cpp scheduler.cpp scheduler_draft.cpp verify_engine.cpp engine.cpp session_manager.cpp draft_runner.cpp mtp_ops.cu
    ../vendor/unicode/unicode.cpp ../vendor/unicode/unicode-data.cpp)
target_include_directories(souris-core PUBLIC ../include PRIVATE . ../vendor ../vendor/unicode ../ggml/src ../ggml/src/ggml-cuda)
target_link_libraries(souris-core PUBLIC ggml ggml-cpu ggml-cuda Threads::Threads CUDA::cudart OpenSSL::Crypto)

if (WIN32)
    target_compile_definitions(souris-core PRIVATE _WIN32_WINNT=0x0602)
    target_link_libraries(souris-core PRIVATE psapi)
endif ()

set_target_properties(souris-core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
add_executable(souris main.cpp server.cpp config.cpp ops_log.cpp)
target_link_libraries(souris PRIVATE souris-core)

target_include_directories(souris PRIVATE ../vendor)

# Native epilogues preserve the arithmetic mode used by their GGML CUDA kernels.
set_source_files_properties(native_ops.cu PROPERTIES COMPILE_OPTIONS "-use_fast_math")
