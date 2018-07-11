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


#calculate iv
def cal_iv(s, k, r, q, T, premium, type):
    tol = 1e-8
    n = 1
    nmax = 100
    sigma_h = math.sqrt(2 * abs((math.log(s/k) + (r-q)*T)/T))
    sigma = sigma_h
    sigmadiff = 1

    if type == 'c':
        while (n < nmax) and (sigmadiff >= tol):
            d1 = cal_d1(s, k, r, q, T, sigma)
            v_vega = cal_vega(s, q, T, d1)
            c_value = cal_eur(s, k, r, q, T, sigma, 'c')
            increment = (c_value - premium) / v_vega
            sigma = sigma - increment
            n = n + 1
            sigmadiff = abs(increment)
        return sigma

    if type == 'p':
        while (n < nmax) and (sigmadiff >= tol):
            d1 = cal_d1(s, k, r, q, T, sigma)
            v_vega = cal_vega(s, q, T, d1)
            p_value = cal_eur(s, k, r, q, T, sigma, 'p')
            increment = (p_value - premium) / v_vega
            sigma = sigma - increment
            n = n + 1
            sigmadiff = abs(increment)
        return sigma

    else:
        return ''

def cal_vega(s, q, T, d1):
    v_vega = s * math.e ** (-q*T) * math.sqrt(T) * sps.norm.pdf(d1)
    return v_vega


#calculate american options
def cal_am(s,k,r,T,sigma,n, type):
    u = math.e ** (sigma* math.sqrt(T/n))
    d = 1/u
    df = math.exp(-r*T/n)
    p = (math.e ** (r*T/n) -d)/(u-d)
    bino_tree = np.zeros((n+1,n+1))
    bino_tree[0][0] = s
    #i-column, j-row, construct tree
    for i in range(n):
        for j in range(i+1):
            bino_tree[j+1][i+1]= d * bino_tree[j][i]
        bino_tree[0][i+1] = u * bino_tree[0][i]
    #calculate the last column
    if type == 'c':
        for j in range(n+1):
            bino_tree[j][n] = max(0,bino_tree[j][n]-k)
        #calculate priemum
        for i in range(n-1,-1,-1):
            for j in range(i+1):
                bino_tree[j][i] = max(df*(p*bino_tree[j][i+1]+(1-p)*bino_tree[j+1][i+1]),bino_tree[j][i]-k)
    if type == 'p':
        for j in range(n + 1):
            bino_tree[j][n] = max(0, k - bino_tree[j][n])
        # calculate priemum
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                bino_tree[j][i] = max(df*(p*bino_tree[j][i+1]+(1-p)*bino_tree[j+1][i+1]),k-bino_tree[j][i])
    return bino_tree[0][0]


#calculate geo-asian options
def cal_geo_asian(s,k,r,T,sigma,n,type):
    sigmahat = sigma * math.sqrt((n+1)*(2*n+1)/(6*n**2))
    miuhat = (r-0.5*sigma**2)*(n+1)/(2*n) + 0.5*sigmahat**2
    d1hat = (math.log(s/k)+(miuhat+0.5*sigmahat**2)*T)/(sigmahat*math.sqrt(T))
    d2hat = (math.log(s/k)+(miuhat-0.5*sigmahat**2)*T)/(sigmahat*math.sqrt(T))
    if type == 'c':
        g_asian_call = math.exp(-r*T)*(s*math.exp(miuhat*T)*sps.norm.cdf(d1hat)-k*sps.norm.cdf(d2hat))
        return g_asian_call
    if type == 'p':
        g_asian_put = math.exp(-r*T)*(k*sps.norm.cdf(-d2hat)-s*math.exp(miuhat*T)*sps.norm.cdf(-d1hat))
        return g_asian_put
    else:
        return ''


#calculate arith-asian options
def cal_arith_asian(s,k,r,T,sigma,n_ob,n_sim,type,cv):
    np.random.seed(0)
#calculate geo-asian parameter
    dt = T/n_ob
    drift = math.exp((r-0.5*sigma**2)*dt)
#calculate stock price
    arith_payoff = []
    geo_payoff = []
    for i in range(n_sim):
        gf = drift * math.exp(sigma*math.sqrt(dt)*np.random.normal(0,1))
        spath = [s*gf]
        for j in range(1,n_ob):
            gf = drift * math.exp(sigma*math.sqrt(dt)*np.random.normal(0,1))
            spath.append(spath[j-1]*gf)
        spath = np.array(spath)
        arith_mean = spath.mean()
        geo_mean = spath.cumprod()[n_ob-1]**(1/n_ob)

        if type == 'c':
            arith_payoff.append(math.exp(-r*T) * max(arith_mean-k,0))
            geo_payoff.append(math.exp(-r*T) * max(geo_mean-k,0))
        else:
            arith_payoff.append(math.exp(-r*T) * max(k-arith_mean, 0))
            geo_payoff.append(math.exp(-r*T) * max(k-geo_mean,0))

    # control variate version
    if cv == 'CV':
        arith_payoff = np.array(arith_payoff)
        geo_payoff = np.array(geo_payoff)
        ari_times_geo = arith_payoff * geo_payoff
        cov = ari_times_geo.mean() - arith_payoff.mean()*geo_payoff.mean()
        theta = cov/geo_payoff.var()
        geo = cal_geo_asian(s, k, r, T, sigma, n_ob, type)
        z = arith_payoff+theta*(geo-geo_payoff)
        z_mean = z.mean()
        z_std = z.std()
        return [z_mean-1.96*z_std/math.sqrt(n_sim),z_mean+1.96*z_std/math.sqrt(n_sim),z_mean]
    else:
        arith_payoff = np.array(arith_payoff)
        z_mean = arith_payoff.mean()
        z_std = arith_payoff.std()
        return [z_mean-1.96*z_std/math.sqrt(n_sim),z_mean+1.96*z_std/math.sqrt(n_sim),z_mean]


#calculate geo-basket
def cal_geo_bskt(s1,s2,sigma1,sigma2,r,T,k,rou,type):
    sigma = math.sqrt(2*sigma1*sigma2*rou+sigma1**2+sigma2**2)/2
    miu = r -0.5*(sigma1**2+sigma2**2)/2 + 0.5*sigma**2
    bg0 = math.sqrt(s1*s2)
    d1hat = (math.log(bg0/k)+(miu+0.5*sigma**2)*T)/(sigma* math.sqrt(T))
    d2hat = (math.log(bg0/k)+(miu-0.5*sigma**2)*T)/(sigma* math.sqrt(T))
    if type == 'c':
        g_bskt_call = math.exp(-r*T)*(bg0*math.exp(miu*T)*sps.norm.cdf(d1hat)-k*sps.norm.cdf(d2hat))
        return g_bskt_call
    if type == 'p':
        g_bskt_put = math.exp(-r*T)*(k*sps.norm.cdf(-d2hat)-bg0*math.exp(miu*T)*sps.norm.cdf(-d1hat))
        return g_bskt_put
    else:
        return ''


#calculate arith-basket
def cal_arith_bskt(s1,s2,sigma1,sigma2,r,T,k,rou,n_sim,type,cv):
    np.random.seed(1)
    drift1 = math.exp((r-0.5*sigma1**2)*T)
    drift2 = math.exp((r-0.5*sigma2**2)*T)
    arith_payoff = []
    geo_payoff = []
#calculate stock price
    for i in range(n_sim):
        ran1 = np.random.normal(0,1)
        ran2 = rou*ran1+math.sqrt(1-rou**2)*np.random.normal(0,1)
        gf1 = drift1 * math.exp(sigma1*math.sqrt(T)*ran1)
        gf2 = drift2 * math.exp(sigma2*math.sqrt(T)*ran2)
        mature_s1 = s1*gf1
        mature_s2 = s2*gf2
        if type == 'c':
            arith_payoff.append(max(((mature_s1+mature_s2)/2-k)*math.exp(-r*T),0))
            geo_payoff.append(max((math.sqrt(mature_s1*mature_s2)-k)*math.exp(-r*T),0))
        else:
            arith_payoff.append(max((k-(mature_s1 + mature_s2)/2)*math.exp(-r*T), 0))
            geo_payoff.append(max((k-math.sqrt(mature_s1*mature_s2))*math.exp(-r*T),0))
    arith_payoff = np.array(arith_payoff)
    # control variate version
    if cv == 'CV':
        geo_payoff = np.array(geo_payoff)
        ari_times_geo = arith_payoff * geo_payoff
        cov = ari_times_geo.mean() - arith_payoff.mean()*geo_payoff.mean()
        theta = cov/geo_payoff.var()
        geo = cal_geo_bskt(s1, s2, sigma1, sigma2, r, T, k, rou, type)
        z = arith_payoff+theta*(geo-geo_payoff)
        z_mean = z.mean()
        z_std = z.std()
        return [z_mean-1.96*z_std/math.sqrt(n_sim),z_mean+1.96*z_std/math.sqrt(n_sim),z_mean]
        #return z_mean
    else:
        arith_payoff = np.array(arith_payoff)
        z_mean = arith_payoff.mean()
        z_std = arith_payoff.std()
        return [z_mean-1.96 * z_std/math.sqrt(n_sim), z_mean+1.96*z_std/math.sqrt(n_sim),z_mean]
