# Looped Transformer

This repo is a fork of `nanochat`, repurposed as a small experimental for **looped transformers** (or recurrent transformers if you prefer).

I saw all the hype around Claude Mythos and one claim about it made me curious: "It may be a recurrent transformer". I started to be curious (and if it powers one of the most powerful llm it must be good) and found this https://ouro-llm.github.io/ (it was not that hard). I found the idea very cool and wanted to experiment it myself, or atleast implement it.

The looped implementation lives in [nanochat/loopedgpt.py](/home/nathan/dev/looped-transformer/nanochat/loopedgpt.py). It is wired into the existing training, evaluation, checkpoint, and inference codepaths so you can compare looped and classical models inside the same harness.

## What Is A Looped Transformer?

Instead of stacking many unique blocks:

```text
Layer1 -> Layer2 -> Layer3 -> ... -> LayerN
```

a looped transformer reuses the same block repeatedly:

```text
Block -> Block -> Block -> Block
```

with shared weights.

Conceptually, the hidden state is refined over internal steps:

```text
h0 -> h1 -> h2 -> h3 -> ... -> hT
```

This fork currently supports:

- repeated application of the same transformer trunk
- logits computed after every loop
- per-loop supervision during training
- a learned exit head for adaptive stopping
- expected-loss training over loop exit probabilities
- entropy regularization on the exit distribution

Why this is interesting:

- a standard transformer gets one forward pass to form an answer
- a looped transformer gets several internal refinement steps
- this trades **parameters** for **compute**
- easy examples can stop early, harder examples can use more internal work

## How Training Works In The Looped Model

The model computes logits after every loop, not just at the end.

The looped forward path is roughly:

```python
h = embed(tokens)

for t in range(num_loops):
    h = block(h)
    logits_t = lm_head(h)
    p_t = sigmoid(exit_head(h[:, -1]))
```

For training, each loop gets its own CE loss:

```python
loss_t = cross_entropy(logits_t, target)
```

The exit head predicts a stop probability `p_t` after each loop. Those probabilities are converted into an exit distribution over exact stopping depths:

```python
q1 = p1
q2 = (1 - p1) * p2
q3 = (1 - p1) * (1 - p2) * p3
```

The final objective is the expected loss over loop depth:

```python
total_loss = sum(q_t * loss_t for q_t, loss_t in zip(q, losses))
```

with entropy regularization:

```python
total_loss = expected_loss - beta * entropy(q)
```

Training uses:

1. per-loop cross-entropy
2. a stopping distribution derived from the exit probabilities
3. expected loss over exit depth
4. entropy regularization to avoid collapse to always-early or always-late exit

NOTE: the exit head is not trained with hand-labeled "correct loop counts". It learns indirectly through the total loss. If stopping early hurts, gradients push early stop probabilities down. If later loops do not help much, gradients push earlier stopping up.

The compatibility rule in this repo is:

- `loss_reduction="mean"` uses the loop-aware objective
- `loss_reduction="none"` preserves the old tokenwise loss contract for existing eval code

## Experiment results

| steps | model | final_bpb | mean_bpb | step_sec | tok/sec | peak_mem_mb |
|---:|---|---:|---:|---:|---:|---:|
| 1k | gpt | 3.0590 | 3.1646 | 0.0395 | 25934.4 | 2003.1 |
|  | looped | 2.8226 | 2.7713 | 0.1188 | 8618.4 | 3806.5 |
| 5k | gpt | 3.0650 | 3.0785 | 0.0400 | 25607.7 | 2003.1 |
|  | looped | 2.7003 | 2.7230 | 0.1050 | 9755.5 | 3312.0 |
| 20k | gpt | 3.1202 | 3.0264 | 0.0396 | 25831.0 | 2003.1 |
|  | looped | 2.5316 | 2.6012 | 0.1007 | 10168.1 | 3312.0 |

To reproduce:
```bash
UV_EXTRA=gpu-cu132 uv run --extra gpu-cu132 python -m scripts.bench_looplm \
--device-type cuda \
--depth 8 \
--sequence-len 512 \
--batch-size 2 \
--num-loops 3 \
--entropy-beta 0.02 \
--exit-threshold 0.7 \
--steps 20000 \
--warmup-steps 5 \
--eval-every 5 \
--num-batches 8 \
--eval-batches 4 \
--window-pattern L
```

**DISCLAIMER**: this test was done at small scale on a single GPU and may absolutely not reflect what happens at large scale.

We can actually observe, if I did not make any mistake in the implementation, that looped models perform better than standard models. This suggests that looping transformer blocks actually make the model stronger with the same parameter budget. \
It is also apparently stronger on MOE models than on dense models: https://arxiv.org/html/2605.09165v1. This could be explained by the fact that at each loop you can use a new set of experts that you did not use before, adding more expert diversity.

### My take

I do not think this is likely to be what is behind Mythos, and the main reason is *inference*.

From a training perspective, this method is very interesting. Looped transformers shift more test-time compute into latent-space refinement before producing the next token. That gives the model several internal passes to improve the representation without increasing the parameter count in the usual way. In practice, that can make a model more capable for a fixed parameter budget, or let a smaller model stay competitive with a larger dense model if the extra internal compute is worth it.

From an inference perspective, though, things get much more awkward. With an adaptive exit mechanism, different sequences in the same batch may want different numbers of loops. That makes serving harder, because compute per token is no longer uniform across the batch. Current inference engines are built around much more regular execution patterns, so supporting looped transformers efficiently would likely require new batching and scheduling logic rather than a small patch on top of existing systems.

The KV-cache handling also gets worse. In a dense transformer, cache state is indexed by layer and token position. In a looped transformer, each loop produces a different internal state, so the cache effectively becomes indexed by loop, layer, and token position. That means the memory footprint of the cache grows with the number of loops, which pushes against the current desire for long context and cheap decoding.

So while this idea is very attractive and have strong advantages, I think the inference stack is currently not ready (at least for millions of users), this represent a way too massive shift for Anthropic.

## Repo Focus

This is not a clean-room recurrent-transformer codebase. It is a pragmatic fork of `nanochat` used for experiments. That means:

- the standard GPT path still exists in [nanochat/gpt.py](/home/nathan/dev/looped-transformer/nanochat/gpt.py)
- the looped model is opt-in through `--model-impl=looped`
- most training and eval scripts are shared between classical GPT and LoopLM
- some repo documentation from upstream still exists in code comments and helper scripts

The useful part is that you can do apples-to-apples comparisons with the same tokenizer, dataloaders, optimizer stack, eval path, and checkpoint format.

## Main Entry Points

### Train a looped model

```bash
UV_EXTRA=gpu-cu132 uv run --extra gpu-cu132 python -m scripts.base_train -- \
  --model-impl=looped \
  --num-loops=2 \
  --entropy-beta=0.01 \
  --exit-threshold=0.5 \
  --depth=8 \
  --num-iterations=100
```

Important flags:

- `--model-impl {gpt,looped}` selects the architecture
- `--num-loops` sets how many recurrent passes the looped model takes
- `--entropy-beta` sets the entropy regularization weight on the loop exit distribution
- `--exit-threshold` sets the inference-time stopping threshold for the exit head

## Looped Speedrun Variant

There is also a looped-model version of the original multi-GPU speedrun:

```bash
bash runs/speedrun_looplm.sh
```

That script mirrors the upstream flow more closely and is meant for larger hardware, not a single 16 GB GPU.

## Benchmarking GPT vs LoopLM

For a direct comparison between the classical and looped implementations, use:

```bash
UV_EXTRA=gpu-cu132 uv run --extra gpu-cu132 python -m scripts.bench_looplm --device-type cuda
```

This benchmark:

- trains both models on the same train batches
- evaluates on held-out validation batches
- reports validation BPB
- keeps the tokenizer, optimizer, and data pipeline fixed

## Inference / KV Cache

Looped inference is integrated into the existing engine path:

- standard GPT uses `KVCache`
- looped GPT uses `LoopedKVCache`

Each loop keeps its own KV state so recurrent passes do not overwrite one another. The implementation is in [nanochat/engine.py](/home/nathan/dev/looped-transformer/nanochat/engine.py).

## Useful Files

- [nanochat/loopedgpt.py](/home/nathan/dev/looped-transformer/nanochat/loopedgpt.py): looped model implementation especially `forward()`
- [scripts/bench_looplm.py](/home/nathan/dev/looped-transformer/scripts/bench_looplm.py): local GPT vs LoopLM benchmark
- [runs/speedrun_16gb.sh](/home/nathan/dev/looped-transformer/runs/speedrun_16gb.sh): safer one-GPU run
- [runs/speedrun_looplm.sh](/home/nathan/dev/looped-transformer/runs/speedrun_looplm.sh): looped speedrun variant

## License

MIT. The fork inherits the upstream `nanochat` license.
