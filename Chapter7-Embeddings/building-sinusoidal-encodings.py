# Import PyTorch library for tensor operations
import torch

# Step 1: Set up configuration parameters
embedding_dim = 6
max_sequence_length = 11

# Step 2: Create a matrix to hold all positional encodings
positional_encodings = torch.zeros(max_sequence_length, embedding_dim)

# Step 3: Create position values [0, 1, 2, ..., max_sequence_length-1]
positions = torch.arange(max_sequence_length, dtype=torch.float).reshape(max_sequence_length, 1)

# Step 4: Get dimension indices for even dimensions (these are the 'i' values)
even_dim_indices = torch.arange(0, embedding_dim, 2)

# Step 5: Get dimension indices for odd dimensions
odd_dim_indices = torch.arange(1, embedding_dim, 2)

# Step 6: Compute denominators for even dimensions
even_denominators = torch.pow(10000, even_dim_indices / embedding_dim)

# Step 7: Compute denominators for odd dimensions
odd_denominators = torch.pow(10000, (odd_dim_indices - 1) / embedding_dim)

# Step 8: Compute sine encodings for even dimensions
even_encodings = torch.sin(positions / even_denominators)

# Step 9: Compute cosine encodings for odd dimensions
odd_encodings = torch.cos(positions / odd_denominators)

# Step 10: Interleave sine and cosine encodings
stacked_encodings = torch.stack([even_encodings, odd_encodings], dim=2)

# Step 11: Flatten to create final interleaved pattern
final_positional_encodings = torch.flatten(stacked_encodings, start_dim=1, end_dim=2)

print(f"Positional encodings shape: {final_positional_encodings.shape}")
