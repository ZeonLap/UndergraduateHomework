import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, *args):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensors = args

class Relu(Layer):
	def __init__(self, name):
		super(Relu, self).__init__(name)

	def forward(self, input):
		self._saved_for_backward(input)
		return np.maximum(0, input)

	def backward(self, grad_output):
		input, = self._saved_tensors
		return grad_output * (input > 0)

class Sigmoid(Layer):
	def __init__(self, name):
		super(Sigmoid, self).__init__(name)

	def forward(self, input):
		output = 1 / (1 + np.exp(-input))
		self._saved_for_backward(output)
		return output

	def backward(self, grad_output):
		output, = self._saved_tensors
		return grad_output * output * (1 - output)

class Selu(Layer):
    def __init__(self, name):
        super(Selu, self).__init__(name)

    def forward(self, input):
        # TODO START
        lmd = 1.0507
        alpha = 1.67326
        output = lmd * np.where(input > 0, input, alpha * (np.exp(input) - 1))
        self._saved_for_backward(output)
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        output,  = self._saved_tensors
        lmd = 1.0507
        alpha = 1.67326
        # Using output to avoid exp calculation
        return grad_output * np.where(output > 0, lmd, output + alpha * lmd)
        # TODO END

class Swish(Layer):
    def __init__(self, name):
        super(Swish, self).__init__(name)

    def forward(self, input):
        # TODO START
        sigmoid_u = 1 / (1 + np.exp(-input))
        output = input * sigmoid_u
        self._saved_for_backward(sigmoid_u, output)
        return output
        # TODO END

    def backward(self, grad_output):
        # TODO START
        # f'(u) = (u * sigmoid(u))'
        #       = sigmoid(u) + u * sigmoid(u) * (1 - sigmoid(u))
        #       = sigmoid(u) + f(u) * (1 - sigmoid(u)))
        #       = f(u) + sigmoid(u) * (1 - f(u))
        sigmoid_u, output = self._saved_tensors
        return grad_output * (output + sigmoid_u * (1 - output))
        # TODO END

class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)

    def forward(self, input):
        # TODO START
        h = np.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * input ** 3))
        output = 0.5 * input * (1 + h)
        self._saved_for_backward(input, h)
        return output
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        c = np.sqrt(2 / np.pi)
        alp = 0.044715
        input, h = self._saved_tensors
        return grad_output * (0.5 * (1 + h + input * (1 - h ** 2) * (c + 3 * c * alp * input ** 2)))
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        return input @ self.W + self.b
        # TODO END

    def backward(self, grad_output):
        # TODO START
        X, = self._saved_tensors
        self.grad_W = X.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0)
        return grad_output @ self.W.T
        # TODO END

    def update(self, config):
        mm = config.momentum
        lr = config.learning_rate
        wd = config.weight_decay

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
