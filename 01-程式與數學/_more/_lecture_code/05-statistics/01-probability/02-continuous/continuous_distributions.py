import scipy.stats

# --- 常態分佈 (Normal Distribution) ---
def calculate_normal_probability_between(x1: float, x2: float, mu: float, sigma: float) -> float:
    """
    計算常態分佈在 x1 和 x2 之間的機率。

    參數:
    x1 (float): 區間下限。
    x2 (float): 區間上限。
    mu (float): 平均數。
    sigma (float): 標準差。

    回傳:
    float: 在區間 [x1, x2] 內的機率。
    """
    if sigma <= 0:
        raise ValueError("標準差 sigma 必須大於 0")
    
    # 使用累積分布函數 (CDF) 來計算區間機率
    # P(x1 < X < x2) = CDF(x2) - CDF(x1)
    prob_less_than_x2 = scipy.stats.norm.cdf(x=x2, loc=mu, scale=sigma)
    prob_less_than_x1 = scipy.stats.norm.cdf(x=x1, loc=mu, scale=sigma)
    
    return prob_less_than_x2 - prob_less_than_x1

# --- 指數分佈 (Exponential Distribution) ---
def calculate_exponential_probability_greater_than(x: float, lam: float) -> float:
    """
    計算指數分佈在 x 之後的機率。

    參數:
    x (float): 臨界點。
    lam (float): 速率參數 (lambda)，等於平均值的倒數。

    回傳:
    float: 大於 x 的機率。
    """
    if lam <= 0:
        raise ValueError("速率參數 lambda 必須大於 0")
        
    # P(X > x) = 1 - CDF(x)
    return 1 - scipy.stats.expon.cdf(x=x, scale=1/lam)
    
# 執行範例
if __name__ == "__main__":
    # 常態分佈範例：某地區成年男性的身高平均為 175 公分，標準差為 7 公分。
    # 隨機抽取一位男性，其身高介於 170 到 180 公分之間的機率是多少？
    height_prob = calculate_normal_probability_between(170, 180, mu=175, sigma=7)
    print(f"身高介於 170 至 180 公分的機率為: {height_prob:.4f}")

    # 指數分佈範例：某客服中心平均每小時接到 5 通電話，
    # 距離上一通電話超過 0.5 小時 (30 分鐘) 才接到下一通電話的機率是多少？
    # 平均每小時 5 通電話 -> 速率參數 lam = 5
    call_prob = calculate_exponential_probability_greater_than(0.5, lam=5)
    print(f"超過 30 分鐘才接到電話的機率為: {call_prob:.4f}")