import math

import torch
import numpy as np

# Functions to apply AWGN when NO normalization happens to signal

def snr_db_to_sigma_signal_no_normalize(x, snr_db_signal=None):
    if snr_db_signal is None :
        return None

    snr_linear_signal_no_normalize = 10 ** (snr_db_signal/10.0)
    signal_power_no_normalize = x.pow(2).mean(dim=1, keepdim=True)
    noise_power_no_normalize = signal_power_no_normalize/ snr_linear_signal_no_normalize
    sigma_signal = torch.sqrt(noise_power_no_normalize)
    return sigma_signal


def awgn_channel_no_normalize(x, sigma_signal=None,snr_db_signal=None):
    if snr_db_signal is not None:
        sigma_signal = snr_db_to_sigma_signal_no_normalize(x, snr_db_signal)

    if sigma_signal is None: # For now it doesnt consider the case where sigma passed is negative
        return x

    noise_signal = torch.randn_like(x, device=x.device) * sigma_signal
    return x + noise_signal


# Functions to apply AWGN when normalization happens to signal

def rms_power_normalize_signal(x, eps=1e-8):
    signal_power = x.pow(2).mean(dim=1, keepdim=True)
    return x /torch.sqrt(signal_power + eps)

def snr_db_to_sigma_signal_normalize(snr_db_signal):
    snr_linear_signal_normalize = 10 ** (snr_db_signal/10.0)
    sigma_signal = math.sqrt(1.0 / snr_linear_signal_normalize)
    return sigma_signal

def awgn_channel_normalize(x, sigma_signal=None, snr_db_signal=None):
    if snr_db_signal is not None:
        sigma_signal = snr_db_to_sigma_signal_normalize(snr_db_signal)

    if sigma_signal is None:
        return x

    noise_signal = torch.randn_like(x, device=x.device) * sigma_signal
    return x + noise_signal

# Functions to apply AWGN to model updates when no normalization happens
# Since model updates can be matrices, kernels, etc. mean across dim=1 doesn't make sense
#  Treating the model weight matrix as one flat vector

# --- Helpers ---

def _flatten_dict(w):
    keys = list(w.keys())
    w_flat = torch.cat([w[k].flatten().float() for k in keys])
    return w_flat, keys


def _unflatten_dict(noisy_w_flat, w, keys):
    noisy_w_dict = {}
    idx = 0
    for k in keys:
        numel = w[k].numel()
        noisy_w_dict[k] = noisy_w_flat[idx:idx + numel].reshape(w[k].shape)
        idx += numel
    return noisy_w_dict

# --- No normalize ---

def snr_db_to_sigma_weight_no_normalize(w_flat, snr_db_agg=None):
    if snr_db_agg is None:
        return None

    snr_linear_weight_no_normalize = 10 ** (snr_db_agg / 10.0)
    weight_power_no_normalize = w_flat.pow(2).mean()
    noise_power_no_normalize = weight_power_no_normalize / snr_linear_weight_no_normalize
    sigma_agg = torch.sqrt(noise_power_no_normalize)
    return sigma_agg

def awgn_agg_no_normalize(w, sigma_agg=None, snr_db_agg=None):
    if sigma_agg is None and snr_db_agg is None:
        return w

    w_flat, keys = _flatten_dict(w)

    if snr_db_agg is not None:
        sigma_agg = snr_db_to_sigma_weight_no_normalize(w_flat, snr_db_agg)

    noise = torch.randn_like(w_flat) * sigma_agg
    noisy_w_flat = w_flat + noise

    return _unflatten_dict(noisy_w_flat, w, keys)

# --- Normalize ---

def rms_power_normalize_weight(w_flat, eps=1e-8):
    weight_power = w_flat.pow(2).mean()
    return w_flat / torch.sqrt(weight_power + eps)

def snr_db_to_sigma_weight_normalize(snr_db_agg):
    snr_linear_weight_normalize = 10 ** (snr_db_agg / 10.0)
    sigma_agg = math.sqrt(1 / snr_linear_weight_normalize)
    return sigma_agg

def awgn_agg_normalize(w, sigma_agg=None, snr_db_agg=None):
    if sigma_agg is None and snr_db_agg is None:
        return w

    w_flat, keys = _flatten_dict(w)

    if snr_db_agg is not None:
        sigma_agg = snr_db_to_sigma_weight_normalize(snr_db_agg)

    noise = torch.randn_like(w_flat) * sigma_agg
    noisy_w_flat = w_flat + noise

    return _unflatten_dict(noisy_w_flat, w, keys)