import math
import scipy.stats as sps
import numpy as np

#b.s model
def cal_d1(s, k, r, q, T, sigma):
    v_d1 = (math.log(s/k) + (r-q)*T) / (sigma * math.sqrt(T)) \
           + 0.5*sigma* math.sqrt(T)
    return v_d1

def cal_d2(s, k, r, q, T, sigma):
    v_d2 = (math.log(s/k) + (r-q)*T) / (sigma * math.sqrt(T)) \
           - 0.5*sigma* math.sqrt(T)
    return v_d2

#use b.s to calculate call &put
def cal_eur(s, k, r, q, T, sigma, type):
    d1 = cal_d1(s, k, r, q, T, sigma)
    d2 = cal_d2(s, k, r, q, T, sigma)

    if type == 'c':
        c_value = s * math.e ** (-q*T) * sps.norm.cdf(d1) \
                  - k * math.e ** (-r*T) * sps.norm.cdf(d2)
        return c_value
    if type == 'p':
        p_value = k * math.e ** (-r * T) * sps.norm.cdf(-d2) \
                  - s * math.e ** (-q * T) * sps.norm.cdf(-d1)
        return p_value
    else:
        return ''

