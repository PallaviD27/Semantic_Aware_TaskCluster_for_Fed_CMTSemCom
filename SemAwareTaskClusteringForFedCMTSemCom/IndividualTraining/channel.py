import torch


def awgn_channel(x, sigma):
    if sigma is None or sigma <=0:
        return x
    else:
        noise = torch.randn_like(x, device=x.device) * sigma
        return x + noise
