import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma **2) * T ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

def main():
    st.title("Interactive Black-Scholes Option Pricing Model")
    st.write("Welcome! This model helps you explore the pricing of European call and put options using the Black-Scholes model. Visualize option prices with changing stock prices and volatility through heatmaps.")

    st.sidebar.header("Set Your Parameters")
    st.sidebar.write("Adjust values to see how they impact the option prices!")

    S = st.sidebar.number_input("Current Asset Price (S)", value=100.0, min_value=0.01)
    K = st.sidebar.number_input("Strike Price (K)", value=100.0, min_value=0.01)
    T = st.sidebar.number_input("Maturity (T) in Years", value=1.0, min_value=0.0)
    r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, min_value=0.0, max_value=1.0, step=0.01)

    if S <= 0 or K <= 0 or T < 0 or sigma <= 0:
        st.error("Please make sure S, K, and sigma are positive values, and T is non-negative.")
        return

    call_price = black_scholes(S, K, T, r, sigma, option_type='call')
    put_price = black_scholes(S, K, T, r, sigma, option_type='put')

    st.write(f"### Calculated Option Prices:")
    st.write(f"**Estimated Call Price:** ${call_price:.2f}")
    st.write(f"**Estimated Put Price:** ${put_price:.2f}")

    spot_steps = 10
    vol_steps = 10

    S_min = S * 0.8
    S_max = S * 1.2
    sigma_min = sigma * 0.5
    sigma_max = sigma * 1.5

    S_range = np.linspace(S_min, S_max, spot_steps)
    sigma_range = np.linspace(sigma_min, sigma_max, vol_steps)

    S_grid, sigma_grid = np.meshgrid(S_range, sigma_range)

    call_prices = np.zeros_like(S_grid)
    put_prices = np.zeros_like(S_grid)

    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            call_prices[i, j] = black_scholes(
                S_grid[i, j], K, T, r, sigma_grid[i, j], option_type='call'
            )
            put_prices[i, j] = black_scholes(
                S_grid[i, j], K, T, r, sigma_grid[i, j], option_type='put'
            )

    sns.set(style="whitegrid")

    fmt = ".2f"
    x_labels = [f"{x:.2f}" for x in S_range]
    y_labels = [f"{y:.2f}" for y in sigma_range]

    fig1, ax1 = plt.subplots(figsize=(8,6))
    sns.heatmap(call_prices, annot=True, fmt=fmt,
                xticklabels=x_labels,
                yticklabels=y_labels,
                cmap="viridis", cbar_kws={'label': 'Call Price'}, ax=ax1)
    ax1.set_title('Call Prices Heatmap')
    ax1.set_xlabel('Stock Price (S)')
    ax1.set_ylabel('Volatility (σ)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.write("### Call Option Prices Heatmap")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(put_prices, annot=True, fmt=fmt,
                xticklabels=x_labels,
                yticklabels=y_labels,
                cmap="viridis", cbar_kws={'label': 'Put Price'}, ax=ax2)
    ax2.set_title('Put Prices Heatmap')
    ax2.set_xlabel('Stock Price (S)')
    ax2.set_ylabel('Volatility (σ)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.write("### Put Option Prices Heatmap")
    st.pyplot(fig2)

    st.write("---")
    st.markdown(
        "Created by Mateusz Jastrzębski  |   [LinkedIn](https://www.linkedin.com/in/mateusz-jastrz%C4%99bski-8a2622264/) | [GitHub](https://github.com/MateuszJastrzebski21)")

main()
