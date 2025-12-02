import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        
        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_seq_len, d_model)
        
        # Create position tensor [0, 1, 2, ..., max_seq_len-1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create the division term for the formula
        # For even dimensions: 10000^(2i/d_model) where i ranges from 0 to d_model/2-1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a trainable parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Add positional encoding to the input
        x = x + self.pe[:x.size(1), :]
        return x
