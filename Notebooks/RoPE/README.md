# **RoPE (Rotary Position Embedding)**

Implemented the go to architectural tweak for position encoding used in contemporary LLMs. RoPE overcomes the drawbacks of the alternative position encoding techniques like sine, absolute and relative embeddings. Unlike absolute embeddings, the RoPE is position invariant i.e., the attention score between two embeddings that are same distance or positions apart regardless of their absolute positions will be the same in entire sequence, and it is also more efficient than relative embeddings. 

Here, each embedding is rotated by theta times a scaling factor of its position. 


> Ex. ["The", "less", "you", "know", "the", "better"]  
> Token embeddings for "The" is rotated by 0 * theta  
> Token embeddings for "less" is rotated by 1 * theta  
> Token embeddings for "you" is rotated by 2 * theta

This is to simplify the real working, in reality, each pair of embedding dimensions is rotated by different angle. theta_i = 10000 ^ (-2i/d).

i ranges from 0 to d/2, which represents current vector pair and d is number of dimensions.

## **Length Generalization Stress Test**

To compare the performance of the model over unseen or unwitnessed context length of data, I trained a model with absolute position embeddings and model with RoPE, on 13 random books from Project Gutenberg with following configuration and data:

- Training tokens: 768464
- Model context length: 4096
- Train sequence length: 256
- Epochs: 2
- Batch size: 2
- Embedding size: 768
- Number of heads: 12
- Number of layers: 12
- Vocabulary size: 50257


The model (RoPE) took 1006.65s to train.
<br/>
The model (absolute) took 999.06s to train.


### **Learning Curves**

![RoPE](./rope%20loss.png)
![Absoulte](./abs%20loss.png)

### **Findings**

Since RoPE encodes position by rotating each embedding which results in all embeddings being unique without explicit learnable embedding layer like in absolute position embedding, the position information learned based on the inner product between tokens. And the inner product depends on the relative angles between embeddings irrespective of absolute angle, so RoPE model performs well even upon scaling the context length that model hasn't witnessed. The perplexity will remain stable across increasing context length. On the other hand, absolute position embedding learns weight parameters based on the context length it witnessed during the training, so, upon increasing sequence length the perplexity would also increase.

Above is the empirical observation, but in my experiment even though the model converge faster in RoPE and perplexity remained lower than abs, there's no sign of increasing perplexity in abs with increase in context length.

| Context Length | RoPE Perplexity | Abs Perplexity |
| :--- | :--- | :--- |
| 256 | 218.529 | 335.048 |
| 512 | 227.305 | 348.660 |
| 1024 | 242.533 | 358.215 |
| 1536 | 252.937 | 365.744 |
| 2048 | 255.161 | 364.280 |
| 4096 | 265.132 | 366.783 |

![Length Generalization Stress Test](./length%20generalization%20stress%20test.png)

