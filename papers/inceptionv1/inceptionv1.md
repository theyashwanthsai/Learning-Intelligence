# Paper #3 – LeNet-5 (1998)

**Paper Going deeper with convolutions :** [Link](https://arxiv.org/pdf/1409.4842)  

**Implementation:** [Code Link](./inceptionv1.ipynb)

---

### 0. Input

Shape: 224 × 224 × 3 (height × width × colour channels).
This is the RGB image you feed in.

### 1. First “stem” – turning raw pixels into low-level features

#### Architecture

| Step | Operation                                 | Output Shape   | What it does                                                                                              |
| ---- | ----------------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------- |
| 1    | **7×7 Convolution, 64 filters, stride 2** | 112 × 112 × 64 | Large “wide-angle” filters that detect simple edges/colours while shrinking the image by half (stride 2). |
| 2    | **3×3 Max-Pool, stride 2**                | 56 × 56 × 64   | Keeps the strongest edge responses and halves the size again.                                             |
| 3    | **1×1 Convolution, 64 filters**           | 56 × 56 × 64   | Learns a new mix of the 64 channels—like re-colouring the feature maps.                                   |
| 4    | **3×3 Convolution, 192 filters**          | 56 × 56 × 192  | Detects slightly more complex patterns from the re-coloured maps.                                         |
| 5    | **3×3 Max-Pool, stride 2**                | 28 × 28 × 192  | Reduces spatial size; we now have a “compressed but information-rich” representation.                     |

### 2. Inception Modules – the key novelty

Here the network branches in **parallel**.
At each module, the same 28×28 or 14×14 feature map is fed simultaneously to four different “experts”, each looking at a different context:

- 1×1 conv path – looks at each pixel location individually (no surrounding neighbourhood) but can recombine channel information cheaply.

- 1×1 → 3×3 conv path – first reduces channel count (saves compute), then looks at a small neighbourhood.

- 1×1 → 5×5 conv path – same idea but a larger neighbourhood.

- 3×3 max-pool → 1×1 conv path – pooling for a very coarse view, followed by a 1×1 conv to tidy up.

All four outputs are concatenated depth-wise, like stacking layers of paper into a thicker ream.
This lets the network decide which scale of feature is useful at that stage.

#### First Inception stage (size 28×28)

- 3a: outputs 256 channels (mix of all four branches).

- 3b: outputs 480 channels.

Then a 3×3 Max-Pool shrinks size to 14×14.

#### Second Inception stage (size 14×14)

Five modules in sequence, gradually widening the channel dimension while the spatial grid stays 14×14:

4a → 4b → 4c → 4d → 4e
(final one outputs 832 channels).

Auxiliary classifier #1 is attached after 4a and #2 after 4d during training.
These are small side networks (avg-pool → 1×1 conv → FC → softmax) whose loss helps gradients flow back to early layers.

Then another 3×3 Max-Pool cuts the map to 7×7.

#### Third Inception stage (size 7×7)

Two more modules:

- 5a → outputs 832 channels

- 5b → outputs 1024 channels.

Now each “pixel” in this 7×7 grid summarises a fairly large patch of the original image.

### 3. Classifier Head

Global Average Pool over the 7×7 grid → gives 1×1×1024 (one number per channel, the average activation).
This replaces giant fully connected layers used in older nets.

Dropout (p = 0.4) for regularisation.

Fully Connected layer → 1000-way softmax for ImageNet classes.

## Intuition

- Earlier CNNs (AlexNet, VGG) were straight stacks of layers with one filter size at a time.

- GoogLeNet’s Inception modules are like mini-ensembles inside the network:

    - Some “look small” (1×1), some “look medium” (3×3), some “look wide” (5×5), plus a pooling scout.

    - Their results are fused so the next layer sees a rich multi-scale description.

    - 1×1 convolutions before 3×3 or 5×5 cut the number of input channels, which keeps computation and parameters low.

    - Auxiliary classifiers keep gradients healthy in this 22-layer deep model.

## References
- [Original paper](https://arxiv.org/pdf/1409.4842)
- [Theoretical explanation]()