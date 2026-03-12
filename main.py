import math
from dataclasses import dataclass

def norm_pdf(x: float) -> float:
    return (1.0/ math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, sigma and T must all be positive")
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

def d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
    return d1(S, K, r, sigma, T) - sigma * math.sqrt(T)


def call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    return S * norm_cdf(D1) - K * math.exp(-r * T) * norm_cdf(D2)


def put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    return K * math.exp(-r * T) * norm_cdf(-D2) - S * norm_cdf(-D1)


def call_delta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    return norm_cdf(d1(S, K, r, sigma, T))


def put_delta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return -1.0 if S < K else 0.0
    return norm_cdf(d1(S, K, r, sigma, T)) - 1.0

def gamma(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return 0.0
    D1 = d1(S, K, r, sigma, T)
    return norm_pdf(D1) / (S * sigma * math.sqrt(T))

def vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return 0.0
    D1 = d1(S, K, r, sigma, T)
    return S * norm_pdf(D1) * math.sqrt(T)

def call_theta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return 0.0
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    first_term = -(S * norm_pdf(D1) * sigma) / (2.0 * math.sqrt(T))
    second_term = -r * K * math.exp(-r * T) * norm_cdf(D2)
    return first_term + second_term

def put_theta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return 0.0
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    first_term = -(S * norm_pdf(D1) * sigma) / (2.0 * math.sqrt(T))
    second_term = r * K * math.exp(-r * T) * norm_cdf(-D2)
    return first_term + second_term

def call_rho(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return 0.0
    D2 = d2(S, K, r, sigma, T)
    return K * T * math.exp(-r * T) * norm_cdf(D2)

def put_rho(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return 0.0
    D2 = d2(S, K, r, sigma, T)
    return -K * T * math.exp(-r * T) * norm_cdf(-D2)

@dataclass
class OptionInputs:
    S: float = 100.0
    K: float = 100.0
    T: float = 30 / 252
    r: float = 0.05
    sigma: float = 0.20

def price_and_greeks(inputs: OptionInputs) -> dict:
    S = inputs.S
    K = inputs.K
    T = inputs.T
    r = inputs.r
    sigma = inputs.sigma

    return {
    'call price': call_price(S, K, r, sigma, T),
    'put price': put_price(S, K, r, sigma, T),
    'call delta': call_delta(S, K, r, sigma, T),
    'put delta': put_delta(S, K, r, sigma, T),
    'gamma': gamma(S, K, r, sigma, T),
    'vega': vega(S, K, r, sigma, T),
    'call theta': call_theta(S, K, r, sigma, T),
    'put theta': put_theta(S, K, r, sigma, T),
    'call rho': call_rho(S, K, r, sigma, T),
    'put rho': put_rho(S, K, r, sigma, T),
}

if __name__ == '__main__':
    inputs = OptionInputs(
        S = 100,
        K = 100,
        T = 30 / 252,
        r = 0.05,
        sigma = 0.20
    )

    results = price_and_greeks(inputs)
    print('Black-Scholes options pricer with greek risk metrics')
    print('-' * 40)
    print(f'Stock price (S): {inputs.S}')
    print(f'Strike price (K): {inputs.K}')
    print(f'Risk-free Rate (r): {inputs.r}')
    print(f'volatility (sigma): {inputs.sigma}')
    print(f'Time to expiry (T): {inputs.T:.4f} years')
    print('-' * 40)

    for key, value in results.items():
        print(f'{key:15s}: {value:.6f}')