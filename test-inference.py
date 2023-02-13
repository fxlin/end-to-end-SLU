#!/usr/bin/env python3.6
# xzl

import data
import models
import soundfile as sf
import torch

from torchinfo import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config("experiments/no_unfreezing.cfg"); _,_,_=data.get_SLU_datasets(config)
print('config load ok')
model = models.Model(config).eval()
model.load_state_dict(torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device)) # load trained model

print(model)  # print model arch

signal, _ = sf.read("test2.wav")
signal = torch.tensor(signal, device=device).float().unsqueeze(0) # xzl: make 1-size batch

print(signal.size())  # e.g. [1,47554]
#intent_dummy = torch.LongTensor(1,3); intent_dummy.zero_() # batch 1, slot 3
intent_dummy = torch.LongTensor([[0,0,0]]) # batch 1, slot 3

summary(model, signal.size(), y_intent=intent_dummy)       # whole model

# xzl: attempted, no success
# summary(model.pretrained_model, signal.size(), y_phoneme=torch.LongTensor(1,3150,42), y_word=torch.LongTensor(1,19,256))

intent = model.decode_intents(signal)
print(intent)


