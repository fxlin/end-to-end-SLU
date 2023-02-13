#!/usr/bin/env python3.6
# xzl
# https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

import soundfile as sf
import torch

from torchinfo import summary


rnn = torch.nn.GRU(10, 20, 2)  # input_size, hidden_size, #layers
input = torch.randn(5, 3, 10) # length, batchsize, input_size
h0 = torch.randn(2, 3, 20)    # numlayers, batchsize, hiddensize
output, hn = rnn(input, h0)

print(output.size(), hn.size())

summary(rnn, input.size())

