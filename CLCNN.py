import torch
import torch.nn as nn
from torchvision import models

class CLCNN(nn.Module):
	def __init__(self):
		super(CLCNN, self).__init__()
		m = models.vgg16(pretrained=True)
		# m.load_state_dict(torch.load('pth/vgg16-397923af.pth'))
		self.encoder = nn.Sequential(*list(m.children())[:-1])
		self.projection = nn.Sequential(nn.Linear(25088, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, 128, bias=True))
		self.classifier = nn.Sequential(*list(m.children())[-1], nn.Linear(1000, 8), nn.Softmax(dim=1))

		# self.plt = nn.Linear(128,2)
		# for param in self.plt.parameters():
		# 	param.requires_grad = False


	def forward(self, x):
		x = self.encoder(x)
		x = x.reshape(x.shape[0], -1)
		repr = self.projection(x)
		x = self.classifier(x)
		# plt = self.plt(repr)
		return x, repr#	, plt