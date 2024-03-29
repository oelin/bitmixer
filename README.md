# BitMixer

An implementation of BitMixer in PyTorch. BitMixer is an MLP-Mixer architecture that uses `tanh` for soft [binarization](https://arxiv.org/abs/1602.02830). During inference, the model's weights and activations can be quantized to 1-bit.

Also, check out this recent paper exploring binarized transformers: [https://arxiv.org/abs/2310.11453](https://arxiv.org/abs/2310.11453).
