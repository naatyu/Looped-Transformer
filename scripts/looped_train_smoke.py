import argparse
import gc

import torch

from nanochat.loopedgpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import HuggingFaceTokenizer, get_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Tiny LoopLM training smoke test")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-kv-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--num-loops", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--use-data", action="store_true", help="load a real batch from the downloaded parquet sample")
    parser.add_argument("--data-split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--entropy-beta", type=float, default=0.01)
    parser.add_argument("--exit-threshold", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    cfg = GPTConfig(
        sequence_len=args.sequence_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        window_pattern="L",
        num_loops=args.num_loops,
        entropy_beta=args.entropy_beta,
        exit_threshold=args.exit_threshold,
    )

    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    model.train()

    opt = model.setup_optimizer(
        unembedding_lr=1e-3,
        embedding_lr=1e-3,
        matrix_lr=1e-3,
        weight_decay=0.0,
        scalar_lr=1e-3,
    )

    if args.use_data:
        try:
            tokenizer = get_tokenizer()
        except FileNotFoundError:
            print("local tokenizer missing; falling back to gpt2 tokenizer for smoke test")
            tokenizer = HuggingFaceTokenizer.from_pretrained("gpt2")
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer,
            args.batch_size,
            args.sequence_len,
            split=args.data_split,
            device=device,
        )
        x, y = next(loader)
    else:
        x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.sequence_len), device=device)
        y = torch.randint(0, cfg.vocab_size, (args.batch_size, args.sequence_len), device=device)

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss = model(x, y, loss_reduction="mean")
        print(f"step={step} loss={float(loss.detach()):.6f}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            token_loss = model(x, y, loss_reduction="none")
            print(f"step={step} token_loss_shape={tuple(token_loss.shape)}")

    if device.type == "cuda":
        torch.cuda.synchronize()
        print(f"max_mem_mb={torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")

    del model, opt, x, y, loss, token_loss
    gc.collect()


if __name__ == "__main__":
    main()
