import pytest
from probability_models import (
    calculate_binomial_probability,
    calculate_bernoulli_probability,
    calculate_poisson_probability
)

# --- 二項分佈測試 (Binomial Distribution Tests) ---
def test_binomial_coin_toss():
    n, k, p = 3, 2, 0.5
    assert calculate_binomial_probability(n, k, p) == pytest.approx(0.375)

def test_binomial_edge_case():
    n, k, p = 4, 0, 0.2
    assert calculate_binomial_probability(n, k, p) == pytest.approx(0.4096)

# --- 伯努利分佈測試 (Bernoulli Distribution Tests) ---
def test_bernoulli_success():
    """測試成功機率"""
    p = 0.7
    assert calculate_bernoulli_probability(1, p) == pytest.approx(0.7)

def test_bernoulli_failure():
    """測試失敗機率"""
    p = 0.7
    assert calculate_bernoulli_probability(0, p) == pytest.approx(0.3)

def test_bernoulli_invalid_k():
    """測試 k 值不合法"""
    with pytest.raises(ValueError, match="必須是 0 或 1"):
        calculate_bernoulli_probability(2, 0.5)

# --- 泊松分佈測試 (Poisson Distribution Tests) ---
def test_poisson_basic():
    """
    測試泊松分佈的基本案例。
    預期結果: PMF(k=2, lambda=3) = (e^-3 * 3^2) / 2! ≈ 0.22404
    """
    k, lam = 2, 3.0
    assert calculate_poisson_probability(k, lam) == pytest.approx(0.22404, rel=1e-4)

def test_poisson_zero_events():
    """
    測試發生 0 次事件的機率。
    預期結果: PMF(k=0, lambda=1.5) = e^-1.5 ≈ 0.22313
    """
    k, lam = 0, 1.5
    assert calculate_poisson_probability(k, lam) == pytest.approx(0.22313, rel=1e-4)

def test_poisson_invalid_lam():
    """測試 lambda 值不合法"""
    with pytest.raises(ValueError, match="必須大於 0"):
        calculate_poisson_probability(2, 0)