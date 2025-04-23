import torch

if torch.cuda.is_available():
    print("Using GPU: ", torch.cuda.get_device_name(0))
else:
    print("Not using GPU! Using CPU.")