# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 22:03:07 2025

@author: Utilisateur
"""
import numpy as np
from scipy.optimize import brentq
from math import exp
from .bs_pricer import call_price_bs, put_price_bs

def _bounds_price(option_type: str, S: float, K: float, r: float, T: float):
    discK = K * exp(-r * T)
    if option_type == "call":
        lower = max(0.0, S - discK)
        upper = S
    else:
        lower = max(0.0, discK - S)
        upper = discK
    return lower, upper

def implied_vol(price: float, S: float, K: float, r: float, T: float, option_type: str = "call",
                vol_lower: float = 1e-6, vol_upper: float = 5.0) -> float:
    option_type = option_type.lower()
    lb, ub = _bounds_price(option_type, S, K, r, T)
    if price < lb - 1e-12 or price > ub + 1e-12:    
        return np.nan
    def f(sig):
        if sig <= 0:
            return (lb - price)
        if option_type == "call":
            return call_price_bs(S, K, r, sig, T) - price
        else:
            return put_price_bs(S, K, r, sig, T) - price
        try:
            vol = brentq(f, vol_lower, vol_upper, maxiter=200, xtol=1e-12)
            return float(vol)
        except Exception:
            return np.nan