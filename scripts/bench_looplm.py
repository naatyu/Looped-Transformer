import argparse
import gc
import time
from dataclasses import asdict

import torch

from nanochat.common import autodetect_device_type, compute_init, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import HuggingFaceTokenizer, get_tokenizer
from nanochat.tokenizer import get_token_bytes
from nanochat.gpt import GPT as BaseGPT, GPTConfig as BaseGPTConfig
from nanochat.loopedgpt import GPT as LoopedGPT, GPTConfig as LoopedGPTConfig


def build_tokenizer():
    try:
        return get_tokenizer()
    except FileNotFoundError:
        print0("local tokenizer missing; falling back to gpt2 tokenizer for benchmark")
        return HuggingFaceTokenizer.from_pretrained("gpt2")


def collect_batches(tokenizer, batch_size, seq_len, split, device, num_batches):
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        batch_size,
        seq_len,
        split=split,
        device=device,
    )
    batches = []
    for _ in range(num_batches):
        x, y = next(loader)
        batches.append((x, y))
    return batches


def build_model(kind, cfg, device):
    if kind == "gpt":
        model_cls = BaseGPT
        config_cls = BaseGPTConfig
        config_kwargs = asdict(cfg)
    else:
        model_cls = LoopedGPT
        config_cls = LoopedGPTConfig
        config_kwargs = asdict(cfg)
        config_kwargs["num_loops"] = cfg.num_loops
        config_kwargs["entropy_beta"] = cfg.entropy_beta
        config_kwargs["exit_threshold"] = cfg.exit_threshold

    with torch.device("meta"):
        model = model_cls(config_cls(**config_kwargs))
    model.to_empty(device=device)
    model.init_weights()
    model.train()
    return model


def eval_model(model, batches, token_bytes, device):
    model.eval()
    try:
        with torch.no_grad():
            return float(evaluate_bpb(model, batches, len(batches), token_bytes))
    finally:
        model.train()


def run_benchmark(model, train_batches, eval_batches, token_bytes, warmup_steps, steps, eval_every, device):
    optimizer = model.setup_optimizer(
        unembedding_lr=1e-3,
        embedding_lr=1e-3,
        matrix_lr=1e-3,
        weight_decay=0.0,
        scalar_lr=1e-3,
    )

    def train_step(x, y):
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        loss = model(x, y, loss_reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return float(loss.detach()), time.perf_counter() - t0

    for step in range(warmup_steps):
        x, y = train_batches[step % len(train_batches)]
        train_step(x, y)

    train_losses = []
    times = []
    peak_mem = 0.0
    eval_history = []

    for step in range(steps):
        x, y = train_batches[step % len(train_batches)]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        loss, dt = train_step(x, y)
        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device) / 1024 / 1024)
        train_losses.append(loss)
        times.append(dt)
        if eval_every > 0 and (((step + 1) % eval_every == 0) or (step + 1 == steps)):
            eval_bpb = eval_model(model, eval_batches, token_bytes, device)
            eval_history.append((step + 1, eval_bpb))

    return {
        "train_losses": train_losses,
        "final_train_loss": train_losses[-1],
        "mean_train_loss": sum(train_losses) / len(train_losses),
        "eval_history": eval_history,
        "final_eval_bpb": eval_history[-1][1] if eval_history else float("nan"),
        "mean_eval_bpb": sum(v for _, v in eval_history) / len(eval_history) if eval_history else float("nan"),
        "mean_step_sec": sum(times) / len(times),
        "tokens_per_sec": (train_batches[0][0].numel() * steps) / sum(times),
        "peak_mem_mb": peak_mem,
    }


def main():
    parser = argparse.ArgumentParser(description="Bench LoopLM vs classical GPT on matched local training")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu (empty = autodetect)")
    parser.add_argument("--data-split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-len", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--num-loops", type=int, default=2)
    parser.add_argument("--entropy-beta", type=float, default=0.01)
    parser.add_argument("--exit-threshold", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--window-pattern", type=str, default="L")
    parser.add_argument("--vocab-size", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)

    tokenizer = build_tokenizer()
    vocab_size = args.vocab_size or tokenizer.get_vocab_size()
    token_bytes = get_token_bytes(device=device)
    train_batches = collect_batches(tokenizer, args.batch_size, args.sequence_len, "train", device, args.num_batches)
    eval_batches = collect_batches(tokenizer, args.batch_size, args.sequence_len, "val", device, args.eval_batches)

    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim

    common_kwargs = dict(
        sequence_len=args.sequence_len,
        vocab_size=vocab_size,
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )

    models = {}
    torch.manual_seed(args.seed)
    models["gpt"] = build_model("gpt", BaseGPTConfig(**common_kwargs), device)
    torch.manual_seed(args.seed)
    models["looped"] = build_model(
        "looped",
        LoopedGPTConfig(
            **common_kwargs,
            num_loops=args.num_loops,
            entropy_beta=args.entropy_beta,
            exit_threshold=args.exit_threshold,
        ),
        device,
    )

    results = {}
    for name, model in models.items():
        print0(f"Running {name}...")
        results[name] = run_benchmark(model, train_batches, eval_batches, token_bytes, args.warmup_steps, args.steps, args.eval_every, device)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print0("")
    print0("Benchmark summary")
    print0("name    final_bpb   mean_bpb    step_sec   tok/sec   peak_mem_mb")
    for name in ["gpt", "looped"]:
        r = results[name]
        print0(
            f"{name:<7} {r['final_eval_bpb']:10.4f} {r['mean_eval_bpb']:10.4f} "
            f"{r['mean_step_sec']:9.4f} {r['tokens_per_sec']:9.1f} {r['peak_mem_mb']:11.1f}"
        )

    delta = results["looped"]["final_eval_bpb"] - results["gpt"]["final_eval_bpb"]
    print0("")
    print0(f"final_bpb_delta(looped-gpt) = {delta:+.4f}")


if __name__ == "__main__":
    main()
