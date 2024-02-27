# BitMixer

An implementation of BitMixer in PyTorch. 

BitMixer is a MLP-Mixer architecture that uses `tanh` for soft [binarized](https://arxiv.org/abs/1602.02830). During inference, the model's weights and activations can be quantized to 1-bit, allowing for efficient hardware implementations.
