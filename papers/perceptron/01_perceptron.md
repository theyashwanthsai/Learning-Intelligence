# Paper #1 – Perceptron (1958)

**Paper:** [Link](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf)  
**Authors:** Rosenblatt  
**Dataset:** Toy binary dataset  
**Implementation:** [Code Link](./perceptron.py)

---

## Architecture
S-points (sensory units).

A-units (association units):

- A₁: Projection area — partially organized connections from S-points, spatially clustered.

- A₂: Association area — randomly connected to A₁ units, performs the main “association” step.

R-units (response units) — receive connections from A₂, decide the output (plus possible feedback connections).

```
Stimulus → S (retina) → A₁ (projection) → A₂ (association) →  R-units (responses)
```

Key Points from the Paper:

- S → A₁: fixed, localized connections (each A₁ sees a small neighborhood in the retina).

- A₁ → A₂: fixed, random connections from A₁.

- A₂ → R: trainable weights (updated via Rosenblatt’s perceptron rule).

- Activation: Binary step (±1).

- Learning: Adjust only A₂ → R weights on misclassification.

## Math
- x = sensory vector (retina input)
- 𝑊𝑆→𝐴1 = fixed weights from retina to projection area (local receptive fields)
- 𝑊𝐴1→𝐴2 = fixed weights from projection to association area
- 𝑊𝐴2→𝑅 = trainable output weights
- 𝑏 = bias
- 𝜂 = learning rate
- sgn(⋅) = sign function (returns +1 if ≥0, else −1)

#### Step 1 (S → A1)

```
a1_input = x.WS->A1

a1_output = sgn(a1_input)
```

#### Step 2 (A1 → A2)

```
a2_input = a1_output.WA1->A2

a2_output = sgn(a2_input)
```

#### Step 3 (A2 → R)

```
r_input = a2_output.WA2->R + b

r_output = sgn(r_input)
```

#### Step 4 (Learning Rule)

```
if r_output != y:
    W_A2->R = W_A2->R + η * y * a2_output
    b = b + η * y
```


## 6. References
- [Original paper](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf)
- [Video 1](https://youtu.be/l-9ALe3U-Fg?si=SPOzBK1dZina8Oud)
- [Video 2](https://youtu.be/Suevq-kZdIw)
