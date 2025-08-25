# My Journey Into the World of Language Models

This repository is my journal for hands-on exploration of building and training large language model from scratch. Before going into fine-tuning and playing with pre-built LLM, I wanted to understand under the hood working of LLMs. Here's what I learned along the way.

## Stages of Journey

### 1. Text Processing & Embeddings
- Built basic tokenizer to convert raw text into model comprehensible token ids.
- Converted token ids to word embeddings to capture semantic relationships between tokens.
- Created dataset for LLM pretraining using small custom text data.

### 2. Attention is all you need
- Implemented self-attention mechanism.
- Progressively built masked attention to focus only on current and previous tokens.
- Created multi-head attention to let the model learn variety of relationships at the same level of details (nouns, verbs, subject, objects, punctuation).

### 3. Building components of GPT-style Model
- Implemented layernorm, shortcut-connection, feedforward neural network.
- Combined these into the core transformer building blocks
- Architected a GPT2-style language model using transformer blocks

### 4. Pretraining the LLM for next-word prediction task
- Pretrained the LLM on prepared dataset to predict the next word based input sequence.
- Implemented temperature scaling and top-k sampling for controlled text generation.
- Successfully loaded and utilized OpenAI's pre-trained GPT2 weights.

### 5. Fine-Tuning for Spam Classification
- Used OpenAI's pre-trained weights.
- Modified LLM architecture making it suitable to the classification task.
- Fine-tuned the model for practical spam classification.

### 6. Supervised Fine-Tuning (Instruction Fine-Tuning)
- Implemented instruction fine-tuning to make the model follow specific commands.
- Explored parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation).
  - Experimented with different LoRA ranks (8, 16, 32, 128) to find optimal balance.
  - Compared performance with and without considering instruction tokens in training loss.

## Key Learnings

> "What I cannot create,  
> \t\t\tI do not understand"  
> \t\t\t\t\t\t- Richard P. Feynman  

Having just an abstract idea of a concept isn't enough to solve a problem fundamentally. With above aim in my mind and freely following the curiosity about workings of a LLM, I started my LLM journey. In this journey, I have tried to piece together each component - from basic tokenization to intricate attention mechanisms to autoregressively generate coherent text - by building them from zero.

### Down the road
Next up in my LLM journey, my focus will be on:
- Preference Alignment
- KV Caching
- Quantization

### Tech Stack
- PyTorch
- Google Colab (for GPU accessibility)