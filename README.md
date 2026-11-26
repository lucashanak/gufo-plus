# Gufo Plus

A fork of [Gufo](https://github.com/gufo-org/gufo) for AMD Strix Halo (`gfx1151`, 128 GiB). It tracks upstream and carries work that has not been merged there yet; finished pieces are sent upstream as pull requests.

## What this fork adds

- **Exact outputs.** Qwen3.8 Flash-Next gives the same tokens regardless of batch size, how a prompt is split into prefill passes, position, context length or timing. A full 149-question suite produces the identical 133,578 tokens on every build since the exactness fixes.
- **GGUF formats.** Every GGML block format opens and dequantizes; Flash-Next loads llama.cpp and Unsloth mixes (IQ1–IQ4, Q2_K–Q6_K, Q4_0/Q4_1/Q5_0/Q5_1, IQ4_NL, Q8_0) in decode, with WMMA prefill for the common expert formats.
- **Flash-Next speed.** Faster prefill (routed experts, fused DeltaNet, 16,384-row passes, WMMA indexer), faster single-user and 8-session decode, confidence-gated MTP chains, and waits that do not spin a CPU core (on Strix Halo a busy core takes package power from the GPU).
- **8-bit K/V that grows.** K/V rows in 8 bits with an F16 scale per 16 values, mapped in 2 MiB chunks as a session grows and capped by `--kv-cache-bytes`: 8 sessions × 262,144 tokens in a 20 GiB budget.
- **BF16 HC inject fix.** Models that store the HC inject rows in BF16 (for example Swift 1.5 GGUFs) produced garbage for prompts of 96+ tokens; inject rows are now widened to F32 at load.
- **Fine-tuned MTP head for Swift 1.5.** Swift ships the base model's MTP head; a head trained on Swift's own outputs raises draft acceptance by 4–6 points (+3–5 % decode) with byte-identical outputs.
- **Serving.** `--reasoning-budget-fraction` (thinking budget as a share of `max_tokens`, with Qwen's wrap-up), model-chosen prefill budget beside active decoders, cache checkpoints that do not split long passes.
- **Qwen3.6-35B-A3B / Ornith 1.5 35B** with native MTP, history drafting and DFlash2 next to MTP (extension `qwen36_35b_a3b`).

## Results

149-question quality suite on one Strix Halo, 8 sessions, MTP:

| Model | IQ | Suite time | Tokens |
| --- | ---: | ---: | ---: |
| Swift 1.5 Qwen3.8 Flash-Next IQ4_XS + fine-tuned MTP head | 95.3 | **1,094 s** | 82,267 |
| Qwen3.8 Flash-Next, v3 recipe | 95.3 | 1,420 s | 133,578 |
| Halogen (closed source), same machine | 95.3 | 1,789 s | 136,589 |

## Upstream status

| PR | Change | State |
| --- | --- | --- |
| [#875](https://github.com/gufo-org/gufo/pull/875) | Prefill rows independent of the pass route | merged |
| [#876](https://github.com/gufo-org/gufo/pull/876) | Dense sweep of pre-budget rows in budget-crossing chains | merged |
| [#877](https://github.com/gufo-org/gufo/pull/877) | Four-token prefill alignment | open |
| [#881](https://github.com/gufo-org/gufo/pull/881) | Idle-spill polling backoff | merged |
| [#882](https://github.com/gufo-org/gufo/pull/882) | Confidence-gated MTP chains | merged |
| [#884](https://github.com/gufo-org/gufo/pull/884) | Q8_0 PLE n-gram tables | merged |
| [#890](https://github.com/gufo-org/gufo/pull/890) | Register-resident routed prefill experts | merged |
| [#893](https://github.com/gufo-org/gufo/pull/893) | 8-bit growing K/V | open |
| [#894](https://github.com/gufo-org/gufo/pull/894) | One-launch decode HC mixer | merged |
| [#895](https://github.com/gufo-org/gufo/pull/895) | Fused prefill DeltaNet | merged |
| [#930](https://github.com/gufo-org/gufo/pull/930) | Every GGML block format in core | merged |
| [#933](https://github.com/gufo-org/gufo/pull/933) | 16,384-row prefill passes with grouped GEMM rasters | merged (passes later back to 4,096 upstream, #954; this fork keeps 16,384) |
| [#950](https://github.com/gufo-org/gufo/pull/950) | BF16 HC inject rows widened to F32 | open |
| [#961](https://github.com/gufo-org/gufo/pull/961) | Wide verification of long prompt-lookup copies | open |
| [#962](https://github.com/gufo-org/gufo/pull/962) | Prefill waits asleep instead of spinning | open |
| [#963](https://github.com/gufo-org/gufo/pull/963) | In-pass grid checkpoints | open |

This branch is rebased onto upstream `main` after the extension migration (#935). Building, running and model documentation are the same as upstream; the original README follows.

---

# Gufo: The ***Open Source*** Strix Halo AI engine 

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

<p align="center">
  <a href="https://github.com/gufo-org/gufo/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/gufo-org/gufo?label=release&amp;logo=github&amp;style=flat&amp;labelColor=141714&amp;color=2f5d3a"></a>
  <a href="https://gufo.blog/docs"><img alt="Documentation" src="https://img.shields.io/badge/docs-gufo.blog-2f5d3a?logo=readthedocs&amp;logoColor=white&amp;style=flat&amp;labelColor=141714&amp;color=2f5d3a"></a>
  <a href="docs/extension-api.md#versions"><img alt="Extension API series" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgufo-org%2Fgufo%2Fmain%2Finclude%2Fgufo%2Fext%2Fapi-series.json&amp;query=%24.minor&amp;prefix=0.&amp;label=extension%20API&amp;logo=cplusplus&amp;style=flat&amp;labelColor=141714&amp;color=2f5d3a"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/github/license/gufo-org/gufo?logo=opensourceinitiative&amp;logoColor=white&amp;style=flat&amp;labelColor=141714&amp;color=2f5d3a"></a>
</p>

Gufo is an open-source inference engine for one machine: AMD Strix Halo, the
Ryzen AI MAX+ 395 with its Radeon 8060S (`gfx1151`) and up to 128 GiB of
unified memory. It serves text, embedding, speech, image and video models over
OpenAI-compatible APIs, from a single binary with kernels tuned for that chip.
It runs on Linux, where releases are built and tested, and natively on
Windows.

Gufo supports one model per use case, the best that fits, and checks each one
against the original before optimizing it. Its cache is built for coding agents
that resend long conversations, and extensions let you bring your own model.

## Models

| Model | What it does | Weights license |
| --- | --- | --- |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next.md) | Text and images | [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) |
| [Qwen3.8 27B](docs/models/qwen3.8-27b.md) | Text and images | Apache 2.0 |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash.md) | Text | MIT |
| [Inkling 2 Mini](docs/models/inkling-2-mini.md) | Text and images | Apache 2.0 |
| [EmbeddingGemma 2](docs/models/embeddinggemma-2.md) | Embeddings for text, images, audio and video | Apache 2.0 |
| [Qwen3-ASR 1.7B](docs/models/qwen3-asr.md) | Speech to text | Apache 2.0 |
| [Qwen3-TTS 1.7B](docs/models/qwen3-tts.md) | Text to speech and voice cloning | Apache 2.0 |
| [Qwen-Image-2.1](docs/models/qwen-image-2.1.md) | Image generation and editing | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), non-commercial |
| [Qwen-Image-2.1-Turbo](docs/models/qwen-image-2.1.md#qwen-image-21-turbo) | Eight-step image generation and editing | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE), non-commercial |
| [MiniMax H3](docs/models/minimax-h3.md) | Video with sound | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) |
| [Loom packages](docs/models/loom.md) | Every [loom-fun](https://huggingface.co/loom-fun) text package on AMD Loom kernels | Each model's own |

Each model page lists the files to download, the modes and the measured speed.
Gufo's own code is MIT; the weights keep their publishers' terms.

Extensions add more models without changing Gufo. With `gufo` set up as in the
[quickstart](#quickstart), install a reviewed one from the
[extension registry](https://github.com/gufo-org/gufo-extension):

```sh
gufo extension install lfm3
```

[Extensions](docs/extensions.md) explains what that runs and how to trust it;
the [catalogue](https://gufo.blog/extensions) lists the official and community
extensions.

## Quickstart

Download Qwen3.8 Flash-Next and its MTP head:

```sh
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision d96201194f103c906994f283f504209dc8d2a10c \
  --include "UD-Q4_K_XL/*" \
  --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --local-dir models/flash-next
```

Define `gufo` to run in the container, with the GPU and the models folder:

```sh
gufo() {
  podman run --rm -it --network host \
    --userns=keep-id:uid=1000,gid=1000 --group-add keep-groups \
    --device /dev/kfd --device /dev/dri --device /dev/accel/accel0 \
    --ulimit memlock=-1 \
    -v ./models:/models:ro \
    -v gufo-extensions:/home/gufo/.local/share/gufo/extensions \
    ghcr.io/gufo-org/toolboxes/gufo-runtime:latest gufo "$@"
}
```

Serve the model, then ask it something from another terminal:

```sh
gufo serve llm --served-model-name flash-next \
  --model /models/flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --speculative mtp --mtp-model /models/flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "flash-next", "messages": [{"role": "user", "content": "Say something"}]}'
```

Any OpenAI client works with the base URL `http://localhost:8080/v1`: Open
WebUI, the OpenAI SDKs, Codex and other agents. If the GPU isn't found, see
[troubleshooting](docs/troubleshooting.md#the-gpu-isnt-found). For the Debian
package, Nix, a source build or Windows, see [install](docs/install.md).

## Documentation

The documentation is on [gufo.blog/docs](https://gufo.blog/docs), one version
per release, and in [`docs/`](docs/README.md) for the development version.

- **Get started:** [install](docs/install.md), [models](docs/models/README.md),
  [command line](docs/cli.md), [server API](docs/server.md).
- **Run it well:** [cache](docs/cache.md),
  [troubleshooting](docs/troubleshooting.md), [benchmarks](docs/benchmarks.md),
  [versions](docs/versions.md).
- **Extend it:** [extensions](docs/extensions.md),
  [writing an extension](docs/writing-extensions.md),
  [extension API](docs/extension-api.md).

The [changelog](CHANGELOG.md) and [releases](https://github.com/gufo-org/gufo/releases)
list what changed.

## Principles

- **One machine.** Every kernel is tuned for Strix Halo with 128 GiB. Smaller
  memory configurations work for the models that fit.
- **The best model per use case.** Fewer models means less code to maintain and
  more time to optimize and test each one. Each model keeps its own kernels, so
  a change to one can't break another.
- **Quality before speed.** Every model has a quality report that compares it
  with an independent reference; an optimization that changes the output isn't
  one.
- **Real workloads.** Concurrent requests, cancellation and long conversations
  are first-class, not afterthoughts.
- **Small production footprint.** Few runtime dependencies; development and
  reference tools stay out of the package.

## Contributing

Contributions are welcome: Gufo needs the Strix Halo community to keep
improving. [CONTRIBUTING.md](CONTRIBUTING.md) explains what belongs in Gufo
and what belongs in an extension. To add a model without changing Gufo,
[write an extension](docs/writing-extensions.md).

Community projects built on Gufo include
[Gufo RDMA for Dual Strix Halo](https://github.com/neuhaus/gufo), which runs
models across two machines and isn't merged yet.

## License

Gufo's original code is [MIT licensed](LICENSE). Its design builds on other
open-source engines, listed in [reference projects](docs/reference-projects.md). Adapted code and dependencies
keep their own notices in [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`, installed
under `share/licenses/gufo`. Model weights aren't bundled and keep their
publishers' terms.
