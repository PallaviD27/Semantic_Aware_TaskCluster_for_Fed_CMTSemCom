
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

# data_dir = './data/mnist/'
# data = datasets.MNIST(data_dir, train= True, download=True, transform=transforms.ToTensor())

y= datasets.mnist.targets.numpy()

print(y[:100])