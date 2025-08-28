# Paper #4 – AlexNet (2012)

**Paper:** [Link](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)  
**Authors:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
**Dataset:** ImageNet
**Implementation:** [Code Link](./alexnet.ipynb)

---

## Architecture
- 8 learned layers:
    - 5 convolutional (Conv) layers → feature extraction
    - 3 fully-connected (FC) layers → classification
- Final layer → 1000-way softmax (for ImageNet’s 1000 classes).
- Uses ReLU after every Conv + FC layer.
- Includes Local Response Normalization (LRN) and Overlapping Max-Pooling in specific spots.
- Trained on two GPUs, with some layers split (I own a rtx 4060, so it should be fine i suppose)

#### Layer-by-layer breakdown
Layer 1 — Conv + ReLU + LRN + MaxPooling
- Conv: 96 filters of size 11 × 11 × 3, stride = 4.
- Output: 55 × 55 × 96 feature maps.
- ReLU applied.
- LRN applied.
- MaxPooling: 3 × 3 window, stride 2 → output size 27 × 27 × 96.

Layer 2 — Conv + ReLU + LRN + MaxPooling
- Conv: 256 filters of size 5 × 5 × 48 (note: split across 2 GPUs).
- Input comes from pooled+normalized Layer 1.
- Output: 27 × 27 × 256 → then pooled → 13 × 13 × 256.
- ReLU applied.
- LRN applied.
- MaxPooling: 3 × 3 window, stride 2 → output 13 × 13 × 256.

Layer 3 — Conv + ReLU
- Conv: 384 filters of size 3 × 3 × 256.
- Connected to all maps from Layer 2 (both GPUs).
- Output: 13 × 13 × 384.
- ReLU applied.
- No pooling, no normalization here.

Layer 4 — Conv + ReLU
- Conv: 384 filters of size 3 × 3 × 192.
- Only connected to half of Layer 3’s maps (per GPU split).
- Output: 13 × 13 × 384.
- ReLU applied.

Layer 5 — Conv + ReLU + MaxPooling
- Conv: 256 filters of size 3 × 3 × 192.
- Output: 13 × 13 × 256 → then pooled → 6 × 6 × 256.
- ReLU applied.
- MaxPooling: 3 × 3 window, stride 2.

Layer 6 — Fully Connected + ReLU + Dropout
- Input: Flattened 6 × 6 × 256 = 9216 values.
- FC: 4096 neurons.
- ReLU applied.
- Dropout (p = 0.5).

Layer 7 — Fully Connected + ReLU + Dropout
- FC: 4096 neurons.
- ReLU applied.
- Dropout (p = 0.5).

Layer 8 — Fully Connected + Softmax
- FC: 1000 neurons.
- Softmax: output probabilities over 1000 ImageNet classes.


Note: LRN isn’t used anymore, it was replaced by Batch Normalization (Ioffe & Szegedy, 2015), which stabilizes training much better


#### Dropout

- During training, it randomly sets a fraction of neurons to 0 (e.g., 50% in FC layers)
- Prevents overfitting (forces the network to not rely on specific neurons)
- Makes the model behave like an ensemble of smaller networks.
- At test time, dropout is turned off, and all neurons are used.


#### Local Response Normalization (LRN)

- This was in the original AlexNet, before BatchNorm became popular
- It tries to mimic "lateral inhibition" in biology: neurons inhibit nearby neurons
- It's not used anymore, it was replaced by Batch Normalization (Ioffe & Szegedy, 2015), which stabilizes training much better


## References
- [Original paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- []()