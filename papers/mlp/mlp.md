# Paper #2 – MLP (2006)

**Paper:** [Link](https://www.cs.utoronto.ca/~hinton/absps/naturebp.pdf)  
**Authors:** D. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
**Dataset:** MNIST
**Implementation:** [Code Link](./mlp.py)

---
To be done:
## Architecture
Not coming from the paper, its just a general architecture for a MLP.

```
Input Layer (64) → Hidden Layer (32) → Output Layer (10)
```

- 64 input neurons — one for each feature in your dataset (input_size=64).
- 1 hidden layer with 32 neurons
- 10 output neurons 

## Math

m -> Batch size

#### Forward Propagation
```
# Layer 1: Input -> Hidden
Z1 = X @ W1 + b1                          # (m x 64) @ (64 x 32) -> (m x 32)
A1 = torch.sigmoid(Z1)                    # activation

# Layer 2: Hidden -> Output
Z2 = A1 @ W2 + b2                         # (m x 32) @ (32 x 10) -> (m x 10)
A2 = torch.sigmoid(Z2)                    # final output

# Loss (Mean Squared Error)
loss = torch.mean((A2 - Y)**2)
```




#### Backward Propagation
```
m = X.shape[0]

# Output layer
dA2 = A2 - Y                               # derivative of MSE wrt A2
dZ2 = dA2 * A2 * (1 - A2)                  # sigmoid derivative
dW2 = (A1.T @ dZ2) / m                     # (32 x m) @ (m x 10) -> (32 x 10)
db2 = torch.sum(dZ2, dim=0, keepdim=True) / m

# Hidden layer
dA1 = dZ2 @ W2.T                           # (m x 10) @ (10 x 32) -> (m x 32)
dZ1 = dA1 * A1 * (1 - A1)                  # sigmoid derivative
dW1 = (X.T @ dZ1) / m                      # (64 x m) @ (m x 32) -> (64 x 32)
db1 = torch.sum(dZ1, dim=0, keepdim=True) / m
```


#### Update
```
# Parameter update (Gradient Descent)
W1 -= lr * dW1
b1 -= lr * db1
W2 -= lr * dW2
b2 -= lr * db2

```


## 6. References
- [Original paper](https://www.cs.utoronto.ca/~hinton/absps/naturebp.pdf)
