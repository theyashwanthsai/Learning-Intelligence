# Paper #4 – AlexNet (2012)

**Paper:** [Link](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)  
**Authors:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
**Dataset:** ImageNet
**Implementation:** [Code Link](./alexnet.ipynb)

---

## Architecture
8 learned layers:
- 5 convolutional (Conv) layers → feature extraction.
- 3 fully-connected (FC) layers → classification.

Final layer → 1000-way softmax (for ImageNet’s 1000 classes).

Uses ReLU after every Conv + FC layer.

Includes Local Response Normalization (LRN) and Overlapping Max-Pooling in specific spots.

Trained on two GPUs, with some layers split.




## References
- [Original paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- []()