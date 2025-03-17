import torch
import torch.nn as nn

# Define a 2D convolutional layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=3, padding=0)

# Input tensor with size (batch_size, in_channels, height, width)
input_tensor = torch.randn(8, 3, 30, 30)  # 8 images, 3 channels, 32x32 pixels

# Apply the convolutional layer
output_tensor = conv_layer(input_tensor)

print(output_tensor.shape)  # Output shape: (8, 16, 32, 32)
