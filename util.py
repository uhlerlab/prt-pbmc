import torch

def torch_random_choice(stuff, size):
  rand_idx = torch.randint(len(stuff), size=(size,))
  return stuff[rand_idx]
