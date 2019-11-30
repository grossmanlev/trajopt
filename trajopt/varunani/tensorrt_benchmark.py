import copy
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet
from trajopt.models.critic_nets import Critic, STATE_DIM


class ModelToTest(object):

    def __init__(self, model, name, x):
        self.model = model
        self.name = name
        self.x = x


if __name__ == '__main__':
    # Create models to test
    models = []
    models.append(
        ModelToTest(model=alexnet(pretrained=True).eval(),
                    name="AlexNet",
                    x=torch.ones((1, 3, 224, 224))))
    models.append(
        ModelToTest(model=Critic(),
                    name="CriticNet",
                    x=torch.ones((1, STATE_DIM))))

    # Evaluate throughput of each non-accelerated, CUDA-accelerated,
    # and TensorRT-accelerated model
    for model in models:
        model_cuda = copy.deepcopy(model.model).cuda()
        x_cuda = model.x.clone().cuda()
        model_trt = torch2trt(model_cuda, [x_cuda])

        print('<=========== {} ===========>'.format(model.name))
        print('Non-accelerated: ')
        for t in tqdm(range(10000)):
            y = model.model(model.x)

        print('CUDA-accelerated: ')
        for t in tqdm(range(10000)):
            y = model_cuda(x_cuda)

        print('TensorRT-accelerated: ')
        for t in tqdm(range(10000)):
            y = model_trt(x_cuda)
