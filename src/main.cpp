#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "config.h"
#include "model.h"
#include "sampler.h"

// USE_CUDA is defined by the `gpu` Makefile target. The two runners share an
// identical interface, so the rest of main() is build-agnostic.
#ifdef USE_CUDA
#include "infer_gpu.h"
using Runner = GpuRunner;
#else
#include "infer_cpu.h"
using Runner = CpuRunner;
#endif

// CLI: tinyllm <model.bin> --ids "785 6722 ..." [--max-new N]
//             [--temp T] [--top-p P] [--seed S] [--dump-logits PATH]
//
// Prints the generated token ids (space-separated) to stdout.
// Diagnostics and tok/s go to stderr.
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.bin> --ids \"...\" "
                     "[--max-new N] [--temp T] [--top-p P] [--seed S] "
                     "[--dump-logits PATH]\n", argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::vector<int> prompt;
    RunConfig cfg;
    std::string dump_logits;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--ids") {
            std::string s = next();
            size_t p = 0;
            while (p < s.size()) {
                while (p < s.size() && s[p] == ' ') ++p;
                if (p >= s.size()) break;
                prompt.push_back(std::atoi(s.c_str() + p));
                while (p < s.size() && s[p] != ' ') ++p;
            }
        } else if (a == "--max-new")     cfg.max_new_tokens = std::atoi(next());
        else if (a == "--temp")          cfg.temperature = (float)std::atof(next());
        else if (a == "--top-p")         cfg.top_p = (float)std::atof(next());
        else if (a == "--seed")          cfg.seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--dump-logits")   dump_logits = next();
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (prompt.empty()) { std::fprintf(stderr, "no --ids given\n"); return 1; }

    Model model(model_path);
    Runner runner(model);
    int V = runner.vocab_size();

    // Prefill: process every prompt token; keep the logits after the last one.
    const float* logits = nullptr;
    for (size_t i = 0; i < prompt.size(); ++i)
        logits = runner.forward(prompt[i], (int)i);

    if (!dump_logits.empty()) {
        FILE* f = std::fopen(dump_logits.c_str(), "wb");
        std::fwrite(logits, sizeof(float), V, f);
        std::fclose(f);
        std::fprintf(stderr, "dumped first-step logits to %s\n",
                     dump_logits.c_str());
    }

    // Generation loop, timed.
    unsigned long long rng = cfg.seed;
    std::vector<int> generated;
    int pos = (int)prompt.size();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        int tok = sample(logits, V, cfg, rng);
        generated.push_back(tok);
        logits = runner.forward(tok, pos++);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    for (size_t i = 0; i < generated.size(); ++i)
        std::printf("%d%s", generated[i],
                    i + 1 < generated.size() ? " " : "\n");

    std::fprintf(stderr, "generated %d tokens in %.3f s  =>  %.2f tok/s\n",
                 (int)generated.size(), secs, generated.size() / secs);
    return 0;
}
