# GPT-2

## Overview

This repository contains a PyTorch implementation of a GPT-style transformer language model, closely modeled after the GPT-124M architecture. The implementation provides a detailed, modular approach to building a transformer-based language model from scratch.

## Model Architecture

The model consists of several key components:

![image](https://github.com/user-attachments/assets/68a5d2d0-8205-4985-8b13-c70852206f2a)

### **Configuration-**
 
- Vocabulary Size: 50,257 tokens
- Context Length: 256 tokens
- Embedding Dimension: 768
- Number of Attention Heads: 12
- Number of Layers: 12

### **Key Components-**
1. **MultiHead attention-** 
   - Implements multi-head self-attention mechanism.
   - Supports causal masking for autoregressive prediction.
   - Includes query, key, and value projections.
   - Applies dropout and scaling.

2. **LayerNorm-**
   - Custom layer normalization implementation.
   - Normalizes input features.
   - Includes learnable scale and shift parameters.

3. **GELU Activation-**
   - Custom GELU (Gaussian Error Linear Unit) activation function.
   - Approximation of the standard GELU activation.

4. **FeedForward-**
   - Two-layer feed-forward network.
   - Expands and then projects back to original dimension.
   - Uses GELU activation.

5. **TransformerBlock-**
   - Combines multi-head attention and feed-forward layers.
   - Includes layer normalization and residual connections.
   - Applies dropout for regularization.

6. **GPTModel-**
   - Top-level model that combines all components.
   - Includes token and positional embeddings.
   - Applies transformer blocks.
   - Generates logits for next token prediction.

## Usage Example 
```python 
# Initialize the model with default configuration
model = GPTModel(GPT_CONFIG_124M)

# Prepare input token indices
input_ids = torch.randint(0, 50257, (batch_size, sequence_length))

# Forward pass
logits = model(input_ids)
```

## Using Custom data
To use this model with Custom data- 
1. Prepare a tokenizer (e.g., using HuggingFace's tokenizers).
2. Tokenize your custom dataset.
3. Create custom configuration if needed.
4. Train or fine-tune the model on your specific data.

Example for custom configuration:
```python
CUSTOM_CONFIG = {
    "vocab_size": len(your_custom_vocab),
    "context_length": 512,  # Adjust as needed
    "emb_dim": 768,
    # ... other parameters
}
custom_model = GPTModel(CUSTOM_CONFIG)
```

## Dependencies
1. PyTorch (torch)
2. torch.nn

## Training Considerations
- The model is designed for next-token prediction.
- Requires a large dataset for training.
- Uses causal (autoregressive) masking in self-attention.

## Limitations
- This is a base implementation and may require additional components for full training.
- Does not include training loop or loss function.
- Designed as an educational and foundational implementation.

## License

[MIT](https://choosealicense.com/licenses/mit/)- Feel free to use, modify, and distribute.
