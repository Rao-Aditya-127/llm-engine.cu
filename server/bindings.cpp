#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "engine.h"

namespace py = pybind11;

PYBIND11_MODULE(llm_engine, m) {
    m.doc() = "TinyLLM pybind11 bindings — Qwen2-0.5B FP16 inference engine";

    py::class_<LLMEngine>(m, "LLMEngine")
        .def(py::init<std::string>(),
             py::arg("model_path"),
             "Load the FP16 weight file and upload weights to the GPU.")

        // Blocking full generation — releases the GIL so other Python
        // threads are not stalled while the GPU runs.
        .def("generate_ids",
             &LLMEngine::generate_ids,
             py::arg("prompt_ids"),
             py::arg("max_tokens"),
             py::arg("temperature") = 0.0f,
             py::arg("top_p")       = 1.0f,
             py::arg("seed")        = 1234ULL,
             py::call_guard<py::gil_scoped_release>(),
             "Run prefill + decode. Returns list of generated token IDs.")

        // Streaming generation — the GIL is re-acquired only when calling
        // back into Python (on_token), not during the CUDA work itself.
        .def("generate_ids_streaming",
             [](LLMEngine&              self,
                const std::vector<int>& prompt_ids,
                int                     max_tokens,
                py::function            on_token,
                float                   temperature,
                float                   top_p,
                unsigned long long      seed) {
                 // Release the GIL before entering the C++ decode loop.
                 py::gil_scoped_release release;
                 self.generate_ids_streaming(
                     prompt_ids, max_tokens,
                     [&on_token](int tok_id) {
                         py::gil_scoped_acquire acquire;  // re-enter Python
                         on_token(tok_id);
                     },
                     temperature, top_p, seed);
             },
             py::arg("prompt_ids"),
             py::arg("max_tokens"),
             py::arg("on_token"),
             py::arg("temperature") = 0.0f,
             py::arg("top_p")       = 1.0f,
             py::arg("seed")        = 1234ULL,
             "Stream tokens: calls on_token(token_id: int) for each decoded token.")

        .def("vocab_size",
             &LLMEngine::vocab_size,
             "Return the model vocabulary size (151936 for Qwen2-0.5B).");
}
