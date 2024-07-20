# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum=0.9, eps=1e-8):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features

		# Parameters
		self.weight = Parameter(torch.randn(num_features))
		self.bias = Parameter(torch.zeros(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		
		# Initialize your parameter
		self.momentum = momentum
		self.eps = eps

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			mean, var = input.mean(dim=0), input.var(dim=0)
			self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
			self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
			return self.weight * (input - mean) / torch.sqrt(var + self.eps) + self.bias
		else:
			return self.weight * (input - self.running_mean) / torch.sqrt(self.running_var + self.eps) + self.bias
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			return input * torch.bernoulli(torch.full_like(input, 1 - self.p, device=input.device)) / (1 - self.p)
		else:
			return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5, in_num=3072, out_num=10, hidden_num=1024):
		super(Model, self).__init__()
		# TODO START
		self.logits = nn.Sequential(
			nn.Linear(in_num, hidden_num),
			BatchNorm1d(hidden_num),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.Linear(hidden_num, out_num)
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
