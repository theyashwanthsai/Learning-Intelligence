# Paper #4 – AlexNet (2012)

**Paper:** [Link](https://arxiv.org/abs/1409.1556)  
**Authors:** Karen Simonyan, Andrew Zisserman
**Dataset:** ImageNet
**Implementation:** [Code Link](./vggnet.ipynb)

---

## Architecture
The key idea behind the VGG architecture is simplicity and depth. The entire network is built using a repeating and uniform block structure.

#### Input
- Image Size: A fixed size of 224X224 pixels.
- Channels: 3 channels for a standard RGB (color) image.
- Preprocessing: The only preprocessing is subtracting the mean RGB value (calculated from the training set) from each pixel.

#### Block 1
Input Shape: 224x224x3

Operations:
- Two consecutive 3x3 convolutional layers with 64 filters each. These scan the image for simple patterns.
- A 2x2 max-pooling layer. This shrinks the feature map, keeping only the most important information.

Output Shape: 112x112x64 (The height and width are halved, and the channel depth is now 64).

#### Block 2
Input Shape: 112x112x64

Operations:
- Two consecutive 3x3 convolutional layers with 128 filters each.
- A 2x2 max-pooling layer.

Output Shape: 56x56x128 (Height/width halved again, channel depth doubled to 128).

#### Block 3
Input Shape: 56x56x128

Operations:
- Three consecutive 3x3 convolutional layers with 256 filters each.
- A 2x2 max-pooling layer.

Output Shape: 28x28x256 (Height/width halved, channel depth doubled to 256).

#### Block 4
Input Shape: 28x28x256

Operations:
- Three consecutive 3x3 convolutional layers with 512 filters each.
- A 2x2 max-pooling layer.

Output Shape: 14x14x512 (Height/width halved, channel depth doubled to 512).

#### Block 5
Input Shape: 14x14x512

Operations:
- Three consecutive 3x3 convolutional layers with 512 filters each.
- A 2x2 max-pooling layer.

Output Shape: 7x7x512 (The final feature map before classification).

#### Classifier
Input Shape: A 7x7x512 feature map.

Operations:
- Flatten: The 7x7x512 map is unrolled into a single long vector of size 25,088 (7 * 7 * 512).
- Fully-Connected Layer 1: Connects the 25,088 inputs to 4096 neurons.
- Fully-Connected Layer 2: Further processes the information with another 4096 neurons.
- Fully-Connected Layer 3 (Output): Reduces the 4096 neurons to 1000 neurons, one for each class in the ImageNet dataset.
- Softmax Function: This function is applied to the 1000 neurons to convert their scores into probabilities, showing the model's confidence for each class.

Final Output: A vector of 1000 probabilities, where the highest value corresponds to the predicted class (e.g., "cat", "dog", "car").

## References
- [Original paper](https://arxiv.org/abs/1409.1556)
- []()