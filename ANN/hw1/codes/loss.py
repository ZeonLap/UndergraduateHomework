from __future__ import division
import numpy as np


class MSELoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        return np.sum((input - target) ** 2) / input.shape[0]
        # TODO END

    def backward(self, input, target):
		# TODO START
        return 2 * (input - target) / input.shape[0]
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        return -np.sum(target * np.log(h)) / input.shape[0]
        # TODO END

    def backward(self, input, target):
        # TODO START
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        return (h - target) / input.shape[0]
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        t = np.sum(input * target, axis=1, keepdims=True)
        return np.sum(np.maximum(0, self.margin - t + input) * (1 - target)) / input.shape[0]
        # TODO END

    def backward(self, input, target):
        # TODO START
        t = np.sum(input * target, axis=1, keepdims=True)
        grad = np.where(self.margin - t + input > 0, 1, 0)
        grad[target == 1] = -np.sum(grad, axis=1) + 1
        return grad / input.shape[0]
        # TODO END


# Bonus
class FocalLoss(object):
    def __init__(self, name, alpha=None, gamma=2.0):
        self.name = name
        if alpha is None:
            self.alpha = [0.1 for _ in range(10)]
        self.gamma = gamma

    def forward(self, input, target):
        # TODO START
        alpha = np.array(self.alpha)
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        return -np.sum((self.alpha * target + (1 - alpha) * (1 - target)) * (1 - h) ** self.gamma * target * np.log(h)) / input.shape[0]
        # TODO END

    def backward(self, input, target):
        # TODO START
        alpha = np.array(self.alpha)
        pre = np.sum((alpha * target + (1 - alpha) * (1 - target)) * target, axis=1, keepdims=True)
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        gt = np.sum(h * target, axis=1, keepdims=True)
        grad_1 = -((1 - h) ** self.gamma) * (-self.gamma * h * np.log(h) + 1 - h)
        grad_2 = -((1 - gt) ** (self.gamma - 1)) * h * (self.gamma * gt * np.log(gt) - (1 - gt))
        grad = pre * np.where(target > 0, grad_1, grad_2)
        return grad / input.shape[0]
        # TODO END
