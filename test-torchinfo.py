#!/usr/bin/env python3.6
from torchinfo import summary
from torchvision import models
import torch

vgg = models.vgg16()
summary(vgg, (1, 3, 224, 224))

m = torch.nn.Conv1d(16,33,3,stride=2)
summary(m, (1,16,200))