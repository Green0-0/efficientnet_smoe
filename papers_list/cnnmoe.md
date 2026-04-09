
⭐ Deep Mixture of Experts via Shallow Embedding https://arxiv.org/abs/1806.01531 
- Kinda old, but might serve as a baseline...
- Also this is sparsity and not MoE so I'm not sure if it applies

✅ CondConv: Conditionally Parameterized Convolutions for Efficient Vision https://arxiv.org/abs/1904.04971
- This is actually equivalent to the Dynamic Convolution but with Sigmoid instead of Softmax
- Problematically, it is a Soft MoE, meaning that there is no sparsity, and all the experts are averaged into the convolution kernel.

✅ Dynamic Convolution: Attention over Convolution Kernels https://arxiv.org/abs/1912.03458
- This is actually equivalent to CondConv but with Softmax instead of Sigmoid

✅ Mixture of Experts in Image Classification: What's the Sweet Spot? https://arxiv.org/abs/2411.18322
- Applied to the pointwise (1x1) CNN and not the depthwise portion, so it might be difficult to interpret 

❓ Routers in Vision Mixture of Experts: An Empirical Study https://arxiv.org/abs/2401.15969
- Only for ViT, but might be applicable for building the router

❓ ParameterNet: Parameters Are All You Need https://arxiv.org/abs/2306.14525
- Kind of useless