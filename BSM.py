# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:28:27 2023

@author: crazy
"""

from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho

# Implementation of Black-Scholes formula in Python
import numpy as np
import pandas as pd
from scipy.stats import norm
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho

# Define variables 
r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30
cptype = 'c'

def blackScholes(r, S, K, T, sigma, type="c"):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        if type == "c":
            price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == "p":
            price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
        return price
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")
        
def delta_calc(r, S, K, T, sigma, type="c"):
    "Calculate delta of an option"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        if type == "c":
            delta_calc = norm.cdf(d1, 0, 1)
        elif type == "p":
            delta_calc = -norm.cdf(-d1, 0, 1)
        return delta_calc
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")
        
def gamma_calc(r, S, K, T, sigma, type="c"):
    "Calculate gamma of a option"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        gamma_calc = norm.pdf(d1, 0, 1)/(S*sigma*np.sqrt(T))
        return gamma_calc
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")
        
def vega_calc(r, S, K, T, sigma, type="c"):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        vega_calc = S*norm.pdf(d1, 0, 1)*np.sqrt(T)
        return vega_calc*0.01
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")
        
def theta_calc(r, S, K, T, sigma, type="c"):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        if type == "c":
            theta_calc = -S*norm.pdf(d1, 0, 1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == "p":
            theta_calc = -S*norm.pdf(d1, 0, 1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
        return theta_calc/365
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")
        
def rho_calc(r, S, K, T, sigma, type="c"):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        if type == "c":
            rho_calc = K*T*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == "p":
            rho_calc = -K*T*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
        return rho_calc*0.01
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")

def calc():
    option_type='p'
    aa = np.round(blackScholes(r, S, K, T, sigma, option_type),3)
    bb = np.round(delta_calc(r, S, K, T, sigma, option_type),3)
    cc = np.round(gamma_calc(r, S, K, T, sigma, option_type),3)
    dd = np.round(vega_calc(r, S, K, T, sigma, option_type),3)
    ee = np.round(theta_calc(r, S, K, T, sigma, option_type),3)
    ff = np.round(rho_calc(r, S, K, T, sigma, option_type),3)
    return aa,bb,cc,dd,ee,ff

r = 0.01
S = 30
K = 40
T = 200/365
sigma = 0.5

def main(r,S,K,T,sigma):
    calc()
    all_df01 = pd.DataFrame({"S":S, "K":K, "R":r, "sigma":sigma, "T":T}, index=[0])
    all_df02 = pd.DataFrame({"Price":calc()[0], "Delta":calc()[1], "Gamma":calc()[2], "Vega":calc()[3], "Theta":calc()[4], "Rho":calc()[5]}, index=[0])
    print(all_df01)
    print(all_df02)
    return 0

main(r,S,K,T,sigma)
