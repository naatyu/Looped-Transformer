# Looped Transformers / Looped Language Models (LoopLM)

Looped Transformers are a variant of Transformers where the same block of layers is reused multiple times.

Instead of increasing model depth by stacking many unique layers:

```text
Layer1 -> Layer2 -> Layer3 -> ... -> LayerN
```

a looped transformer applies the **same block repeatedly**:

```text
Block -> Block -> Block -> Block
```

with shared weights.

This gives the model more effective depth without increasing parameters.

---

## Intuition

A standard transformer gets one forward pass to compute an answer.

A looped transformer gets multiple internal "thinking steps":

```text
h0 -> h1 -> h2 -> h3 -> ... -> hT
```

where:

- `h0` = embeddings
- `h1` = hidden state after 1st loop
- `h2` = hidden state after 2nd loop
- etc.

Each loop refines the representation.

Example:

```text
Question: The capital of France is ___
```

Iteration outputs:

```text
loop1: Paris 0.40
loop2: Paris 0.70
loop3: Paris 0.85
```

The model becomes more confident as it loops.

---

# Forward Pass

Pseudo-code:

```python
h = embed(tokens)

for t in range(T):
    h = transformer_block(h)   # same weights reused
    logits_t = lm_head(h)
```

Unlike a normal transformer, logits can be computed after every loop.

---

# Training Objective

The model is supervised at **every loop**.

For each iteration:

```python
loss_t = cross_entropy(logits_t, target)
```

So if we have 3 loops:

```text
loss1 = 0.92
loss2 = 0.36
loss3 = 0.16
```

Later loops usually perform better.

---

# Adaptive Exit (Stopping Mechanism)

The model learns when to stop computing.

After each loop, a small linear classifier predicts stop probability:

```python
p_t = sigmoid(W @ h_t + b)
```

where:

- `h_t` = hidden state after loop `t`
- `W,b` = learned exit linear parameters

Interpretation:

- high `p_t` → stop now
- low `p_t` → continue looping

Example:

```text
p1 = 0.2
p2 = 0.6
p3 = 0.9
```

---

## Exit Distribution

Raw stop probabilities are converted into probability of exiting **exactly** at step `t`.

Formula:

```python
q1 = p1
q2 = (1-p1) * p2
q3 = (1-p1) * (1-p2) * p3
```

Example:

```text
q1 = 0.20
q2 = 0.48
q3 = 0.288
```

Meaning:

- 20% stop after 1 loop
- 48% stop after 2 loops
- 28.8% stop after 3 loops

---

# Final Loss

Expected loss over all possible exit depths:

```python
total_loss = sum(q_t * loss_t for t in loops)
```

Expanded:

```text
L = q1*loss1 + q2*loss2 + q3*loss3
```

Example:

```text
= 0.2*0.92 + 0.48*0.36 + 0.288*0.16
= 0.403
```

This is the final training loss.

---

# Entropy Regularization

Without regularization, model may collapse to:

- always exit early
- or always use max loops

To prevent collapse:

```python
L_total = expected_loss - beta * entropy(q)
```

where:

```python
entropy(q) = -sum(q_t * log(q_t))
```

This encourages exploration of different depths during training.

---

# How Exit Linear Learns

Important: the exit linear is **not trained with labels** like:

```text
correct exit = 2
```

Instead, it is trained indirectly through gradients from total loss.

If early exit is bad:

```text
loss1 high
loss2 much lower
```

gradient pushes:

```text
p1 down
```

so model keeps computing.

If early exit is already good:

```text
loss1 ~= loss2 ~= loss3
```

gradient pushes:

```text
p1 up
```

so model stops early.

The gate learns:

> "Is another loop worth the compute?"

---

# Training Loop (Full)

```python
losses = []
stop_probs = []

h = embed(tokens)

for t in range(T):
    h = block(h)

    logits = lm_head(h)
    losses.append(cross_entropy(logits, target))

    p = sigmoid(exit_linear(h[:, -1]))
    stop_probs.append(p)

q = stopping_distribution(stop_probs)

loss = sum(q_t * loss_t for q_t, loss_t in zip(q, losses))
loss -= beta * entropy(q)

loss.backward()
```

---

# Why This Helps

Looped transformers trade:

- **parameters** for **compute**

Instead of making model larger:

- keep model smaller
- allow more iterative computation

Benefits:

- better reasoning
- adaptive compute
- parameter efficiency

Observed behavior:

- easy tokens → few loops
- hard tokens → more loops

Example:

```text
2 + 2 = 4        -> stop early
hard math proof  -> use more loops
```

---

# Summary

Looped Transformers:

1. Reuse same transformer block multiple times
2. Compute logits after every loop
3. Train CE loss at every loop
4. Learn stop probability with exit linear
5. Optimize expected loss over exit depths
6. Use entropy regularization to avoid collapse

Core idea:

> Instead of only scaling width/parameters, scale internal computation depth.

This makes the model "think longer" internally before producing output.