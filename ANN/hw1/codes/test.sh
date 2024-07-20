# One hidden layer

# Selu
python run_mlp.py --activation Selu --loss MSELoss                 --name Selu_MSE
python run_mlp.py --activation Selu --loss SoftmaxCrossEntropyLoss --name Selu_Softmax
python run_mlp.py --activation Selu --loss HingeLoss               --name Selu_Hinge
python run_mlp.py --activation Selu --loss FocalLoss               --name Selu_Focal

# Swish
python run_mlp.py --activation Swish --loss MSELoss                 --name Swish_MSE
python run_mlp.py --activation Swish --loss SoftmaxCrossEntropyLoss --name Swish_Softmax
python run_mlp.py --activation Swish --loss HingeLoss               --name Swish_Hinge
python run_mlp.py --activation Swish --loss FocalLoss               --name Swish_Focal

# Gelu
python run_mlp.py --activation Gelu --loss MSELoss                 --name Gelu_MSE
python run_mlp.py --activation Gelu --loss SoftmaxCrossEntropyLoss --name Gelu_Softmax
python run_mlp.py --activation Gelu --loss HingeLoss               --name Gelu_Hinge
python run_mlp.py --activation Gelu --loss FocalLoss               --name Gelu_Focal

# Relu
python run_mlp.py --activation Relu --loss MSELoss                 --name Relu_MSE
python run_mlp.py --activation Relu --loss SoftmaxCrossEntropyLoss --name Relu_Softmax
python run_mlp.py --activation Relu --loss HingeLoss               --name Relu_Hinge
python run_mlp.py --activation Relu --loss FocalLoss               --name Relu_Focal

# Two hidden layers

# Selu
python run_mlp.py --activation Selu --loss MSELoss                 --name Selu_MSE          --hiddenlayers 2
python run_mlp.py --activation Selu --loss SoftmaxCrossEntropyLoss --name Selu_Softmax      --hiddenlayers 2
python run_mlp.py --activation Selu --loss HingeLoss               --name Selu_Hinge        --hiddenlayers 2
python run_mlp.py --activation Selu --loss FocalLoss               --name Selu_Focal        --hiddenlayers 2

# Swish
python run_mlp.py --activation Swish --loss MSELoss                 --name Swish_MSE        --hiddenlayers 2
python run_mlp.py --activation Swish --loss SoftmaxCrossEntropyLoss --name Swish_Softmax    --hiddenlayers 2
python run_mlp.py --activation Swish --loss HingeLoss               --name Swish_Hinge      --hiddenlayers 2
python run_mlp.py --activation Swish --loss FocalLoss               --name Swish_Focal      --hiddenlayers 2

# Gelu
python run_mlp.py --activation Gelu --loss MSELoss                 --name Gelu_MSE          --hiddenlayers 2
python run_mlp.py --activation Gelu --loss SoftmaxCrossEntropyLoss --name Gelu_Softmax      --hiddenlayers 2
python run_mlp.py --activation Gelu --loss HingeLoss               --name Gelu_Hinge        --hiddenlayers 2
python run_mlp.py --activation Gelu --loss FocalLoss               --name Gelu_Focal        --hiddenlayers 2

# Relu
python run_mlp.py --activation Relu --loss MSELoss                 --name Relu_MSE          --hiddenlayers 2
python run_mlp.py --activation Relu --loss SoftmaxCrossEntropyLoss --name Relu_Softmax      --hiddenlayers 2
python run_mlp.py --activation Relu --loss HingeLoss               --name Relu_Hinge        --hiddenlayers 2
python run_mlp.py --activation Relu --loss FocalLoss               --name Relu_Focal        --hiddenlayers 2