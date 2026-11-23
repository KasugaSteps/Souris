#include "config.h"
#include "server.h"
#include "souris/runtime.h"
#include "souris/engine.h"
#include "ops_log.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstdio>
#include <csignal>
#include <iostream>
#include <string>

namespace {
void fatal(const char * message) {
    const auto line = nlohmann::json{{"error", message}}.dump();
    std::fwrite(line.data(), 1, line.size(), stdout); std::fputc('\n', stdout); std::fflush(stdout);
}
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int) { interrupted = 1; }
int chat(souris::Runtime & runtime, souris::OpsLog & log) {
    const auto previous = std::signal(SIGINT, interrupt);
    std::string history, line;
    uint64_t sequence = 0;
    while (std::cerr << "> " && std::getline(std::cin, line)) {
        interrupted = 0;
        std::string error;
        auto lease = runtime.requests().acquire(souris::RequestGate::Clock::now() + souris::RequestGate::admission_wait,
            [] { return interrupted != 0; }, error);
        if (!lease) { std::cerr << error << '\n'; continue; }
        auto & sessions = runtime.session(*lease);
        sessions.open("cli");
        if (line == "/exit") { lease->finish([&] { sessions.close(); }); break; }
        if (line == "/reset") {
            lease->finish([&] { sessions.reset(); }); history.clear(); continue;
        }
        const auto prompt = history + line + "\n";
        souris::CompletionOptions options;
        options.max_tokens = std::min(512, sessions.context_limit() - int(sessions.token_count(prompt)));
        const auto result = sessions.generate(prompt, options, [](const std::string & text) {
            std::cerr << text << std::flush; return !interrupted && bool(std::cerr);
        }, [] { return interrupted != 0; });
        std::cerr << '\n';
        lease->finish([&] {
            if (result.completed) { sessions.commit(); history = prompt + result.text; }
            else { sessions.abandon(); std::cerr << result.error << '\n'; }
            log.request_end(sessions, result, "cli-" + std::to_string(++sequence), result.error);
        });
    }
    std::signal(SIGINT, previous);
    return 0;
}
}
int main(int argc, char ** argv) {
    ggml_set_abort_callback(fatal);
    if (argc != 4 || (std::string(argv[1]) != "chat" && std::string(argv[1]) != "serve") || std::string(argv[2]) != "--config") {
        GGML_ABORT("usage: souris chat|serve --config souris.toml");
    }
    auto config = souris::read_config(argv[3]);
    config.serve = std::string(argv[1]) == "serve";
    souris::OpsLog log(config);
    auto runtime = souris::Runtime::build(config);
    std::fprintf(stderr, "%s\n", runtime->engine().startup().c_str());
    return config.serve ? souris::serve(*runtime, log) : chat(*runtime, log);
}
