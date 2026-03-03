import scipy.stats
import math

# --- 二項分佈 (Binomial Distribution) ---
def calculate_binomial_probability(n: int, k: int, p: float) -> float:
    """
    計算二項分佈的機率 (PMF)。
    適用於 n 次獨立試驗中，剛好 k 次成功的機率。
    """
    if not (0 <= k <= n):
        raise ValueError("成功的次數 k 必須介於 0 和 n 之間")
    if not (0 <= p <= 1):
        raise ValueError("機率 p 必須介於 0 和 1 之間")
    
    return scipy.stats.binom.pmf(k, n, p)

# --- 伯努利分佈 (Bernoulli Distribution) ---
def calculate_bernoulli_probability(k: int, p: float) -> float:
    """
    計算伯努利分佈的機率 (PMF)。
    適用於單次試驗中，成功 (k=1) 或失敗 (k=0) 的機率。
    """
    if k not in (0, 1):
        raise ValueError("伯努利分佈的結果 k 必須是 0 或 1")
    if not (0 <= p <= 1):
        raise ValueError("機率 p 必須介於 0 和 1 之間")
        
    return scipy.stats.bernoulli.pmf(k, p)

# --- 泊松分佈 (Poisson Distribution) ---
def calculate_poisson_probability(k: int, lam: float) -> float:
    """
    計算泊松分佈的機率 (PMF)。
    適用於在固定時間或空間區間內，發生 k 次事件的機率。
    參數 'lam' (lambda) 是平均事件發生次數。
    """
    if k < 0:
        raise ValueError("事件次數 k 不能為負數")
    if lam <= 0:
        raise ValueError("平均發生次數 lam 必須大於 0")
        
    return scipy.stats.poisson.pmf(k, lam)

# 執行範例
if __name__ == "__main__":
    # 伯努利範例：投擲一個不公平硬幣，正面機率為 0.7，得到正面的機率是多少？
    p_success_bernoulli = 0.7
    prob_bernoulli_success = calculate_bernoulli_probability(1, p_success_bernoulli)
    print(f"投擲不公平硬幣，得到正面的機率為: {prob_bernoulli_success:.4f}")

    # 泊松範例：某網站平均每分鐘有 3 次新註冊，下一分鐘剛好有 2 次新註冊的機率是多少？
    lam_poisson = 3.0
    k_poisson = 2
    prob_poisson = calculate_poisson_probability(k_poisson, lam_poisson)
    print(f"每分鐘平均 3 次新註冊，下一分鐘剛好有 2 次的機率為: {prob_poisson:.4f}")