
import torch
# Step 1: Create two random tensors of shape 3x3
tensor1 = torch.randn(3, 3, requires_grad=True)
tensor2 = torch.randn(3, 3, requires_grad=True)

# Step 2: Perform matrix multiplication between the two tensors
result = torch.matmul(tensor1, tensor2)

# Step 3: Compute gradients using autograd
result.sum().backward()

# The gradients are stored in tensor1.grad and tensor2.grad
print(tensor1.grad)
print(tensor2.grad)

# Step 1: Create a 3x1 tensor and a 1x3 tensor
tensor3 = torch.randn(3, 1)
tensor4 = torch.randn(1, 3)

# Step 2: Use broadcasting to add the two tensors together
result_broadcasted = tensor3 + tensor4

# Step 3: Multiply the result by another 3x3 tensor
tensor5 = torch.randn(3, 3)
result_final = result_broadcasted * tensor5
print(result_final)

# Step 1: Create a tensor of shape (6, 4) using random values and reshape it to shape (3, 8)
tensor6 = torch.randn(6, 4)
tensor7 = tensor6.view(3, 8)

# Step 2: Extract a slice (e.g., the first two columns, all rows)
slice_tensor = tensor7[:, :2]o

print(slice_tensor)
