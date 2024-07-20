# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum=0.9, eps=1e-8):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features

		# Parameters
		self.weight = Parameter(torch.randn(num_features))
		self.bias = Parameter(torch.zeros(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
		self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
		
		# Initialize your parameter
		self.momentum = momentum
		self.eps = eps

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		weight = self.weight.reshape(1, self.num_features, 1, 1)
		bias = self.bias.reshape(1, self.num_features, 1, 1)
		if self.training:
			mean, var = input.mean(dim=(0, 2, 3), keepdim=True), input.var(dim=(0, 2, 3), keepdim=True)
			self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
			self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
			return weight * (input - mean) / torch.sqrt(var + self.eps) + bias
		else:
			return weight * (input - self.running_mean) / torch.sqrt(self.running_var + self.eps) + bias
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			rand_drop = torch.full((input.shape[0], input.shape[1], 1, 1), 1 - self.p, device=input.device, dtype=torch.float32)
			return input * torch.bernoulli(rand_drop) / (1 - self.p)
			# return input * torch.bernoulli(torch.full_like(input, 1 - self.p, device=input.device, dtype=torch.float32)) / (1 - self.p)
		else:
			return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5, in_channels=3, channels=[64, 128], cnn_kern=[3, 5], max_kern=[2, 2]):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.width = 32
		# Calculate the width of the feature map after two convolutional layers
		W = self.width - cnn_kern[0] + 1 + (cnn_kern[0] // 2) * 2
		W = (W - max_kern[0] + (max_kern[0] // 2) * 2) // max_kern[0] + 1
		W = W - cnn_kern[1] + 1 + (cnn_kern[1] // 2) * 2
		W = (W - max_kern[1] + (max_kern[1] // 2) * 2) // max_kern[1] + 1
		self.logits = nn.Sequential(
			# First Convolutional Layer
			nn.Conv2d(in_channels=in_channels, out_channels=channels[0], kernel_size=cnn_kern[0], padding=cnn_kern[0] // 2),
			BatchNorm2d(channels[0]),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.MaxPool2d(kernel_size=max_kern[0], stride=max_kern[0], padding=max_kern[0] // 2),
			# Second Convolutional Layer
			nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=cnn_kern[1], padding=cnn_kern[1] // 2),
			BatchNorm2d(channels[1]),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.MaxPool2d(kernel_size=max_kern[1], stride=max_kern[1], padding=max_kern[1] // 2),
			# Fully Connected Layer
			nn.Flatten(start_dim=1),
			nn.Linear(channels[1] * W * W, 10)
		)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.logits(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
