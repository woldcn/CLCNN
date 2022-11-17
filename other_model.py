import torch
import torch.nn as nn
from torchvision import models

class Xnn(nn.Module):
	def __init__(self, model):
		super(Xnn, self).__init__()
		self.layer1 = model		# true和 false的影响？

		self.repr = nn.Linear(1000,1000)
		self.plt = nn.Linear(1000,2)
		for param in self.plt.parameters():
			param.requires_grad = False
		self.linear = nn.Linear(1000, 8)
		# self.dropout = nn.Dropout()


	def forward(self, x):
		x = self.layer1(x)
		if not torch.is_tensor(x):		# inception_v3
			x = x.logits
		repr = self.repr(x)
		plt = self.plt(repr)
		x = self.linear(repr)
		# x = self.dropout(x)    #####################?
		# x = F.log_softmax(x ,dim=1)
		return x, repr, plt