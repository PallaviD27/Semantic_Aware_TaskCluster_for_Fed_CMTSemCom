import math

import torch
import numpy as np




def rms_power_normalize(x, eps=1e-8):
    power = x.pow(2).mean(dim=1, keepdim=True)
    return x /torch.sqrt(power + eps)

def snr_db_to_sigma(snr_db):
    snr_linear = 10 ** (snr_db/10.0)
    sigma = math.sqrt(1.0 / snr_linear)
    return sigma

def awgn_channel(x, sigma=None, snr_db=None):
    if snr_db is not None:
        sigma = snr_db_to_sigma(snr_db)

    if sigma is None or sigma <=0:
        return x

    noise = torch.randn_like(x, device=x.device) * sigma
    return x + noise