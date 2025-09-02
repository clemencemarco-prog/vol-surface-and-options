# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 23:32:35 2025

@author: Utilisateur
"""

import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

def d1_d2(S: float, K: float, r: float, sigma: float, T : float):
    if T <= 0:
        return np.nan, np.nan
    if sigma <= 0:
        return np.nan, np.nan
    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2
