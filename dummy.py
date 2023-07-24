import torch

inp = torch.load('inp.pt')

module = torch.load('module.pt')

module(inp[0])