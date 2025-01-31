# How to Run the Code

Install PyTorch (if not already installed) by running:
pip install torch

# Explanation of the Code

Matrix Multiplication and Gradient Computation
Creates two 3×33×3 random tensors (tensor1 and tensor2) with requires_grad=True to track gradients.
Performs matrix multiplication using torch.matmul().
Computes gradients by calling .backward() on the summed result.
Prints the gradients stored in tensor1.grad and tensor2.grad.

Broadcasting and Element-wise Multiplication
Creates a 3×13×1 tensor (tensor3) and a 1×31×3 tensor (tensor4).
Uses broadcasting to add them, producing a 3×33×3 result.
Multiplies the result element-wise by another 3×33×3 tensor (tensor5).
Prints the final element-wise multiplied tensor.

Tensor Reshaping and Slicing
Creates a 6×46×4 tensor (tensor6).
Reshapes it into a 3×83×8 tensor (tensor7).
Extracts the first two columns of all rows and prints the result.

# Expected Results
 
The gradients of tensor1 and tensor2 will be printed (random values based on PyTorch’s autograd).
The final element-wise multiplied tensor (result_final) will contain random values.
The reshaped and sliced tensor will display only the first two columns from tensor7.

