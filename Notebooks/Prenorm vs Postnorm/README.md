# **Prenorm vs Postnorm**

Earlier transformers architecture including the original from the paper, "Attention Is All You Need", implemented postnorm. But later, prenorm was introduced with the argument that postnorm required training warmup for stability and it was difficult to train for models with deep layers.

In postnorm, the layernorm is applied in the main highway, after the residual connection and adding the activations, which resulted into vanishing or exploding gradient during backpropagation through early layers as layernorm scales the values. The gradients compound across the layers, so if the layernorm gradients were even slightly less than 1 it would result into very small gradients in initial layers.

Whereas, in the prenorm, the layernorm is applied in sub-branch before attention mechanism (result -> non-uniform attention scores distribution so model learns quickly) and feed-forward network. Since the prenorm is applied in the sub-branch, the main highway or residual connection has only additive linear path and therefore, the gradients flow smoothly without being scaled.

Also, the gradients in postnorm has high variance across training steps and those spikes instabilizes the training.

## **Experiment**

Implemented and compared the prenorm and postnorm architecture in GPT-2 like LLM. Plotted global gradient norms for both across training steps to observe the gradient spikes.

Based on prior empirical observations, the objective was to witness stable training and smooth gradient norms across training steps in prenorm and vice-versa for postnorm.

Training and model config:

- Training tokens: 34961
- Model context length: 1024
- Train sequence length: 256
- Epochs: 2
- Batch size: 2
- Embedding size: 768
- Number of heads: 12
- Number of layers: 18 (more layers to see the real effect)
- Vocabulary size: 50257

The model (prenorm) took 110.56s to train.
<br/>
The model (postnorm) took 110.01s to train.

### **Findings**

The model training with postnorm resulted into gradient spikes, which unstabilizes the training. This happens because gradients in earlier layers are compounded as layernorm which is applied in main highway (or after residual connection) scales the activations.

![Global Gradient Norm Across Training Steps](./Global%20Gradient%20Norm%20Across%20Training%20Steps.png)