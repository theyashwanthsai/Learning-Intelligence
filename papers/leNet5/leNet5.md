# Paper #3 – LeNet-5 (1998)

**Paper:** [Link](https://axon.cs.byu.edu/~martinez/classes/678/Papers/Convolution_nets.pdf)  
**Authors:** Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner
**Dataset:** Cat vs Dog
**Implementation:** [Code Link](./leNet5.ipynb)

---

## Architecture


LeNet-5 is a **convolutional neural network (CNN)** with **7 layers** (excluding the input). It processes grayscale images of **32×32 pixels**.

```
**LeNet-5 = Conv → Pool → Conv → Pool → Conv → FC → Output**
```

---

```
Architecture overview:

* Input: 32×32
* C1: 28×28×6
* S2: 14×14×6
* C3: 10×10×16
* S4: 5×5×16
* C5: 120
* F6: 84
* Output: 10
```


#### **Layer-by-Layer**

1. **Input layer**

   * Image size: **32×32 pixels** (1 channel, grayscale).

2. **C1 – First Convolutional Layer**

   * 6 feature maps.
   * Kernel size: **5×5**, stride 1.
   * Each output map size: **28×28** (since 32−5+1 = 28).
   * Parameters: **6 × (5×5 + 1) = 156** (including bias).

3. **S2 – First Subsampling (Pooling) Layer**

   * 6 feature maps (same as C1).
   * Each takes a **2×2 average pooling** with a trainable coefficient and bias.
   * Output map size: **14×14**.
   * Parameters: **12** (1 coefficient + 1 bias per map).

4. **C3 – Second Convolutional Layer**

   * 16 feature maps.
   * Kernel size: **5×5**.
   * Output map size: **10×10**.
   * Not all S2 maps are connected to each C3 map (sparse connectivity to reduce parameters and encourage feature diversity).
   * Parameters: **1,516**.

5. **S4 – Second Subsampling (Pooling) Layer**

   * 16 feature maps.
   * Each performs **2×2 average pooling**.
   * Output map size: **5×5**.
   * Parameters: **32**.

6. **C5 – Third Convolutional Layer (Fully Connected Convolution)**

   * 120 feature maps.
   * Kernel size: **5×5**.
   * Each C5 unit is connected to all 16 S4 maps.
   * Output size: **1×1 per feature map → 120 units total**.
   * Parameters: **48,120**.

7. **F6 – Fully Connected Layer**

   * 84 units.
   * Uses the same activation function as previous layers (tanh in the paper).
   * Parameters: **10,164**.

8. **Output Layer**

   * 10 units (for digits 0–9).
   * Uses **Euclidean Radial Basis Function (RBF) units**, though later implementations replaced this with softmax.



## Math
#### Convolution
H_out = (H_in - K + 2*P)/S +1
W_out = (W_in - K + 2*P)/S +1

H_in, W_in -> input height & width
K -> Kernel size
P -> Padding
S -> Stride

#### Pooling
H_out = (H_out - K)/S + 1
W_out = (W_out - K)/S + 1

K -> Pooling kernel size
S -> Stride

## References
- [Original paper](https://axon.cs.byu.edu/~martinez/classes/678/Papers/Convolution_nets.pdf)
- [Visual explanation of how cnns work]()