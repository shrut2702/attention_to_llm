import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
  def __init__(self, dim_in, dim_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert (dim_out % num_heads == 0), "dim_out must be divisible by num_heads"

    self.dim_out = dim_out # final merged context vector embedding size
    self.num_heads = num_heads
    self.head_dim = dim_out//num_heads # embedding size of context vector in single head
    self.w_query = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
    self.w_key = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
    self.w_value = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
    self.out_proj = torch.nn.Linear(dim_out, dim_out) # transform merged context_vectors into similar dimension size vectors
    self.dropout = torch.nn.Dropout(dropout)
    self.register_buffer(
        'mask',
        torch.triu(torch.ones(context_length, context_length), diagonal=1)
    )

  def forward(self, x):
    batch_size, num_tokens, dim_in = x.shape
    queries = self.w_query(x)
    keys = self.w_key(x)
    values = self.w_value(x)  #shape (batch_size, num_tokens, dim_out)

    queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
    keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
    values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim) #shape (batch_size, num_tokens, num_heads, head_dim)

    queries = queries.transpose(1,2)
    keys = keys.transpose(1,2)
    values = values.transpose(1,2) #shape (batch_size, num_heads, num_tokens, head_dim)

    attention_scores = queries @ keys.transpose(2,3)
    attention_scores.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens], -torch.inf)

    attention_weights = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)
    attention_weights = self.dropout(attention_weights)

    context_vectors = (attention_weights @ values).transpose(1,2) #transposing axis 1,2  since we have to merge the context vectors by num_heads and head_dim, so required shape will now be (batch_size, num_tokens, num_heads, head_dim)
    context_vectors = context_vectors.contiguous().view(batch_size, num_tokens, self.dim_out)

    context_vectors = self.out_proj(context_vectors)

    return context_vectors
  
class LayerNorm(nn.Module):
  def __init__(self, emb_size):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_size))
    self.shift = nn.Parameter(torch.zeros(emb_size))

  def forward(self, x):
    mean = x.mean(keepdim=True, dim=-1)
    variance = x.var(keepdim=True, dim=-1, unbiased=False)
    norm_x = (x - mean)/torch.sqrt(variance + self.eps)
    return self.scale * norm_x + self.shift
  
class GeLU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3))
    ))
  
class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(cfg['emb_size'], 4 * cfg['emb_size']),
        GeLU(),
        nn.Linear(4 * cfg['emb_size'], cfg['emb_size'])
    )

  def forward(self, x):
    return self.layers(x)
  
class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.mha = MultiHeadAttention(cfg['emb_size'], cfg['emb_size'], cfg['context_length'], cfg['drop_rate'], cfg['num_heads'], qkv_bias=cfg['qkv_bias'])
    self.layer_norm1 = LayerNorm(cfg['emb_size'])
    self.layer_norm2 = LayerNorm(cfg['emb_size'])
    self.ffn = FeedForward(cfg)
    self.dropout = nn.Dropout(cfg['drop_rate'])

  def forward(self, x):
    shortcut = x
    x = self.layer_norm1(x)
    x = self.mha(x)
    x = self.dropout(x)
    x = x + shortcut

    shortcut = x
    x = self.layer_norm2(x)
    x = self.ffn(x)
    x = self.dropout(x)
    x = x + shortcut

    return x
  
class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.token_emb_layer = nn.Embedding(cfg['vocab_size'], cfg['emb_size'])
    self.pos_emb_layer = nn.Embedding(cfg['context_length'], cfg['emb_size'])
    self.dropout_layer = nn.Dropout(cfg['drop_rate'])
    self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg['num_layers'])]
    )
    self.final_norm = LayerNorm(cfg['emb_size'])
    self.output_layer = nn.Linear(cfg['emb_size'], cfg['vocab_size'], bias=False)

  def forward(self, inp_tokens):
    batch_size, num_tokens = inp_tokens.shape
    token_emb = self.token_emb_layer(inp_tokens)
    pos_emb = self.pos_emb_layer(
        torch.arange(num_tokens, device=inp_tokens.device)
    )
    x = token_emb + pos_emb
    x = self.dropout_layer(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.output_layer(x)

    return logits
  

### COnfiguration for GPT-2 models
BASE_CONFIG = {
    'vocab_size': 50257,
    'context_length': 1024,
    'drop_rate': 0.0,
    'qkv_bias': True
}

model_configs = {
    'gpt2-small (124M)': {
        'emb_size':768,
        'num_heads':12,
        'num_layers':12,
    },
    'gpt2-medium (355M)': {
        'emb_size':1024,
        'num_heads':16,
        'num_layers':24,
    },
    'gpt2-large (774M)': {
        'emb_size':1280,
        'num_heads':20,
        'num_layers':36,
    },
    'gpt2-xl (1558M)': {
        'emb_size':1600,
        'num_heads':25,
        'num_layers':48,
    }
}


### FUnctions to load pre-trained weights
def assign(left, right):
  if left.shape != right.shape:
    raise ValueError(f'Shape mismatch. Left: {left.shape}. Right: {right.shape}')
  return torch.nn.Parameter(torch.tensor(right))

def load_parameters(gpt, params):
  gpt.token_emb_layer.weight = assign(gpt.token_emb_layer.weight, params['wte'])
  gpt.pos_emb_layer.weight = assign(gpt.pos_emb_layer.weight, params['wpe'])

  #transformer blocks' parameters loading
  for b in range(len(params['blocks'])):
    #each blocks' mha qkv weights
    q_w, k_w, v_w = np.split(
        params['blocks'][b]['attn']['c_attn']['w'], 3, axis=-1
    )
    gpt.trf_blocks[b].mha.w_query.weight = assign(
        gpt.trf_blocks[b].mha.w_query.weight, q_w.T
    )
    gpt.trf_blocks[b].mha.w_key.weight = assign(
        gpt.trf_blocks[b].mha.w_key.weight, k_w.T
    )
    gpt.trf_blocks[b].mha.w_value.weight = assign(
        gpt.trf_blocks[b].mha.w_value.weight, v_w.T
    )

    #each blocks' mha qkv bias
    q_b, k_b, v_b = np.split(
        params['blocks'][b]['attn']['c_attn']['b'], 3, axis=-1
    )
    gpt.trf_blocks[b].mha.w_query.bias = assign(
        gpt.trf_blocks[b].mha.w_query.bias, q_b
    )
    gpt.trf_blocks[b].mha.w_key.bias = assign(
        gpt.trf_blocks[b].mha.w_key.bias, k_b
    )
    gpt.trf_blocks[b].mha.w_value.bias = assign(
        gpt.trf_blocks[b].mha.w_value.bias, v_b
    )

    #each blocks' mha out_proj layer's weight and bias
    gpt.trf_blocks[b].mha.out_proj.weight = assign(
        gpt.trf_blocks[b].mha.out_proj.weight, params['blocks'][b]['attn']['c_proj']['w'].T
    )
    gpt.trf_blocks[b].mha.out_proj.bias = assign(
        gpt.trf_blocks[b].mha.out_proj.bias, params['blocks'][b]['attn']['c_proj']['b']
    )

    #each blocks' ff wieghts and biases
    gpt.trf_blocks[b].ffn.layers[0].weight = assign(
        gpt.trf_blocks[b].ffn.layers[0].weight, params['blocks'][b]['mlp']['c_fc']['w'].T
    )
    gpt.trf_blocks[b].ffn.layers[0].bias = assign(
        gpt.trf_blocks[b].ffn.layers[0].bias, params['blocks'][b]['mlp']['c_fc']['b']
    )
    gpt.trf_blocks[b].ffn.layers[2].weight = assign(
        gpt.trf_blocks[b].ffn.layers[2].weight, params['blocks'][b]['mlp']['c_proj']['w'].T
    )
    gpt.trf_blocks[b].ffn.layers[2].bias = assign(
        gpt.trf_blocks[b].ffn.layers[2].bias, params['blocks'][b]['mlp']['c_proj']['b']
    )

    #each blocks' layer_norms scale and shift
    gpt.trf_blocks[b].layer_norm1.scale = assign(
        gpt.trf_blocks[b].layer_norm1.scale, params['blocks'][b]['ln_1']['g']
    )
    gpt.trf_blocks[b].layer_norm1.shift = assign(
        gpt.trf_blocks[b].layer_norm1.shift, params['blocks'][b]['ln_1']['b']
    )
    gpt.trf_blocks[b].layer_norm2.scale = assign(
        gpt.trf_blocks[b].layer_norm2.scale, params['blocks'][b]['ln_2']['g']
    )
    gpt.trf_blocks[b].layer_norm2.shift = assign(
        gpt.trf_blocks[b].layer_norm2.shift, params['blocks'][b]['ln_2']['b']
    )

  #gpt's final_layer_norm
  gpt.final_norm.scale = assign(
        gpt.final_norm.scale, params['g']
  )
  gpt.final_norm.shift = assign(
        gpt.final_norm.shift, params['b']
  )
  gpt.output_layer.weight = assign(
      gpt.output_layer.weight, params['wte']
  )