import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define activation functions
swiglu_fn = lambda x, gate_w, up_w: F.silu(gate_w @ x) * (up_w @ x)
gelu_fn = nn.GELU()

# Generate input data
input_values = torch.linspace(-3, 3, 100)

# For SwiGLU, we need weight matrices (simplified for visualization)
# Using identity-like weights for demonstration
gate_weight = torch.eye(1)
up_weight = torch.eye(1)

# Apply activations
swiglu_output = swiglu_fn(input_values.unsqueeze(0), gate_weight, up_weight).squeeze()
gelu_output = gelu_fn(input_values)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot SwiGLU
axes[0].plot(input_values, swiglu_output, label='SwiGLU', linewidth=2)
axes[0].set_title('SwiGLU Activation Function')
axes[0].set_xlabel('Input (x)')
axes[0].set_ylabel('SwiGLU(x)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot GELU
axes[1].plot(input_values, gelu_output, label='GELU', color='orange', linewidth=2)
axes[1].set_title('GELU Activation Function')
axes[1].set_xlabel('Input (x)')
axes[1].set_ylabel('GELU(x)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()
