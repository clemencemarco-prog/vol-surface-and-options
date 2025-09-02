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
    
def call_price_bs(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
     return max(S - K, 0.0)
    if sigma <= 0:
        forward = S * exp(r * T)
        payoff = max(forward - K, 0.0)
        return payoff * exp(-r * T)
    d1, d2 = d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

def put_price_bs(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        forward = S * exp(r * T)
        payoff = max(K - forward, 0.0)
        return payoff * exp(-r * T)
    d1, d2 = d1_d2(S, K, r, sigma, T)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def greeks_call(S: float, K: float, r: float, sigma: float, T: float):
    if T <= 0 or sigma <= 0:
        return {"delta": float(S > K), "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = d1_d2(S, K, r, sigma, T)
    sqrtT = sqrt(T)
    pdf_d1 = norm.pdf(d1)
    delta = norm.cdf(d1)
    gamma = pdf_d1 / (S * sigma * sqrtT)
    vega = S * pdf_d1 * sqrtT
    theta = (
        - (S * pdf_d1 * sigma) / (2 * sqrtT)
        - r * K * exp(-r * T) * norm.cdf(d2)
    )
    rho = K * T * exp(-r * T) * norm.cdf(d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

def greeks_put(S: float, K: float, r: float, sigma: float, T: float):
    if T <= 0 or sigma <= 0:
        return {"delta": float(S < K) - 1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = d1_d2(S, K, r, sigma, T)
    sqrtT = sqrt(T)
    pdf_d1 = norm.pdf(d1)
    delta = norm.cdf(d1) - 1.0
    gamma = pdf_d1 / (S * sigma * sqrtT)
    vega = S * pdf_d1 * sqrtT
    theta = (
        - (S * pdf_d1 * sigma) / (2 * sqrtT)
        + r * K * exp(-r * T) * norm.cdf(-d2)
    )
    rho = -K * T * exp(-r * T) * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}