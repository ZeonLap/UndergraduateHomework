from network import Network
from utils import LOG_INFO
from layers import Sigmoid, Selu, Swish, Linear, Gelu, Relu
from loss import MSELoss, SoftmaxCrossEntropyLoss, HingeLoss, FocalLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from argparse import ArgumentParser
import time
import wandb

def add_activation(model, config, layer):
    if config.activation == 'Selu':
        model.add(Selu(f'relu{layer}'))
    elif config.activation == 'Swish':
        model.add(Swish(f'swish{layer}'))
    elif config.activation == 'Relu':
        model.add(Relu(f'relu{layer}'))
    elif config.activation == 'Gelu':
        model.add(Gelu(f'gelu{layer}'))
    else:
        raise ValueError('activation must be Selu, Swish, Relu or Gelu')

def init(config):
    model = Network()

    if config.hiddenlayers < 1 or config.hiddenlayers > 3:
        raise ValueError('hiddenlayers must be 1 or 2')
    
    if config.hiddenlayers == 1:
        model.add(Linear('fc1', 784, 128, config.learning_rate))
        add_activation(model, config, 1)
        model.add(Linear('fc2', 128, 10, config.learning_rate))
    elif config.hiddenlayers == 2:
        model.add(Linear('fc1', 784, 256, config.learning_rate))
        add_activation(model, config, 1)
        model.add(Linear('fc2', 256, 64, config.learning_rate))
        add_activation(model, config, 2)
        model.add(Linear('fc3', 64, 10, config.learning_rate))

    # Initialize loss
    if config.loss == 'MSELoss':
        loss = MSELoss('loss')
    elif config.loss == 'SoftmaxCrossEntropyLoss':
        loss = SoftmaxCrossEntropyLoss('loss')
    elif config.loss == 'HingeLoss':
        loss = HingeLoss('loss')
    elif config.loss == 'FocalLoss':
        loss = FocalLoss('loss')
    else:
        raise ValueError('loss must be MSELoss, SoftmaxCrossEntropyLoss, HingeLoss or FocalLoss')
    
    return model, loss


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--activation', choices=['Selu', 'Swish', 'Relu', 'Gelu'], default='Relu')
    parser.add_argument('--loss', choices=['MSELoss', 'SoftmaxCrossEntropyLoss', 'HingeLoss', 'FocalLoss'], default='SoftmaxCrossEntropyLoss')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--disp_freq', type=int, default=50)
    parser.add_argument('--test_epoch', type=int, default=5)
    parser.add_argument('--hiddenlayers', type=int, default=1)

    config = parser.parse_args()

    model, loss = init(config)

    # load data from mnist
    train_data, test_data, train_label, test_label = load_mnist_2d('data')

    wandb.init(project="ANN-HW1",
            config=config,
            name=config.name)

    start = time.time()

    for epoch in range(config.max_epoch):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        train_net(model, loss, config, train_data, train_label, config.batch_size, config.disp_freq, epoch, config.name)

        if epoch % config.test_epoch == 0 or epoch == 99:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_net(model, loss, config, test_data, test_label, config.batch_size, epoch, config.name)
    
    end = time.time()
    wandb.log({f'{config.hiddenlayers}layers_time': end - start})

    wandb.finish()