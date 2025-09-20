import pytest
import math
from continuous_distributions import (
    calculate_normal_probability_between,
    calculate_exponential_probability_greater_than
)

# --- 常態分佈測試 (Normal Distribution Tests) ---
def test_normal_basic_case():
    """
    測試常態分佈的基本案例。
    預期結果：標準常態分佈中，[-1, 1] 區間的機率約為 0.6827
    (68-95-99.7 法則)
    """
    # 參數: x1=-1, x2=1, mu=0, sigma=1
    x1, x2 = -1, 1
    mu, sigma = 0, 1
    
    result = calculate_normal_probability_between(x1, x2, mu, sigma)
    assert result == pytest.approx(0.68268949, rel=1e-5)

def test_normal_z_score():
    """
    測試用 z-score 轉換後的結果是否一致。
    身高 175, 標差 7.5 的情況下，身高介於 170-180 的機率應與標準常態分佈
    z-score 介於 -0.6667 到 0.6667 的機率一致
    """
    mu, sigma = 175, 7.5
    z1 = (170 - mu) / sigma
    z2 = (180 - mu) / sigma
    
    prob_normal = calculate_normal_probability_between(170, 180, mu, sigma)
    prob_standard_normal = calculate_normal_probability_between(z1, z2, 0, 1)
    
    assert prob_normal == pytest.approx(prob_standard_normal)

def test_normal_invalid_sigma():
    """測試標準差 sigma 不合法"""
    with pytest.raises(ValueError, match="必須大於 0"):
        calculate_normal_probability_between(0, 1, 0, 0)

# --- 指數分佈測試 (Exponential Distribution Tests) ---
def test_exponential_basic_case():
    """
    測試指數分佈的基本案例。
    P(X > 1) with lam=1 -> e^-1 ≈ 0.367879
    """
    x, lam = 1, 1
    result = calculate_exponential_probability_greater_than(x, lam)
    assert result == pytest.approx(0.367879, rel=1e-5)

def test_exponential_traffic_jam():
    """
    測試應用情境：某路口平均每 2 分鐘發生 1 次輕微堵車。
    從現在開始超過 5 分鐘後才發生堵車的機率？
    平均每 2 分鐘 1 次 -> lam=1/2 = 0.5
    """
    x, lam = 5, 0.5
    result = calculate_exponential_probability_greater_than(x, lam)
    
    # 預期結果: e^(-0.5 * 5) = e^-2.5 ≈ 0.082085
    assert result == pytest.approx(0.082085, rel=1e-5)

def test_exponential_invalid_lam():
    """測試速率參數 lambda 不合法"""
    with pytest.raises(ValueError, match="必須大於 0"):
        calculate_exponential_probability_greater_than(1, -1)