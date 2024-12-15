import torch

def get_device(use_cpu=False):
    device = torch.device('cuda') if torch.cuda.is_available() and not use_cpu else torch.device('cpu')
    print(f"Device selected: {device}")
    return device

device = get_device()
print(f"Using device: {device}")