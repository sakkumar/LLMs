import torch
import torch.nn.functional as F
from torch.autograd import grad

def forward_pass():
	y = torch.tensor([1.0, 0.0])
	x1 = torch.tensor([[2.0,2.5,1464.0,1.45],[3.0,3.5,2000.0,2.0]])
	w1 = torch.tensor([1.3,2.3,1.2,0.4], requires_grad=True)
	b = torch.tensor([1.0], requires_grad=True)
	z = x1 @ w1 + b
	a = torch.sigmoid(z)
	print("x1: ", x1.shape)
	print("w1: ", w1.shape)
	print("b: ", b.shape)
	loss = F.binary_cross_entropy(a, y)
	print("loss: ", loss)
	loss.backward()
	print("grad_L_w1: ", w1.grad)
	print("grad_L_b: ", b.grad)

if __name__ == "__main__":
    print("forward_pass")
    forward_pass()
