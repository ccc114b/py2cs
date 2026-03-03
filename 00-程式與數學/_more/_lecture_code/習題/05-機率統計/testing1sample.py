import numpy as np
from scipy import stats

def conduct_one_sample_t_test(sample_scores, population_mean, alpha=0.05):
    """
    對一組樣本進行單樣本 t 檢定。

    Args:
        sample_scores (list or numpy.array): 樣本分數。
        population_mean (float): 假設的母體平均數 (虛無假設)。
        alpha (float): 顯著水準。

    Returns:
        tuple: 包含 t 統計量和 p 值。
    """
    # 確保樣本數足夠
    if len(sample_scores) < 2:
        raise ValueError("樣本數必須大於 1")

    # 執行單樣本 t 檢定
    # alternative='two-sided' 表示我們檢定樣本平均數是否「不等於」母體平均數
    t_stat, p_value = stats.ttest_1samp(
        a=sample_scores, 
        popmean=population_mean,
        alternative='two-sided'
    )
    
    return t_stat, p_value

def make_decision(p_value, alpha=0.05):
    """
    根據 p 值和顯著水準做出檢定決策。

    Args:
        p_value (float): t 檢定得到的 p 值。
        alpha (float): 顯著水準，通常為 0.05。

    Returns:
        str: 檢定結果的文字描述。
    """
    if p_value < alpha:
        return f"p 值 ({p_value:.4f}) 小於顯著水準 ({alpha})，有足夠證據拒絕虛無假設。\n結論：樣本平均數與假設的母體平均數有顯著差異。"
    else:
        return f"p 值 ({p_value:.4f}) 大於顯著水準 ({alpha})，沒有足夠證據拒絕虛無假設。\n結論：樣本平均數與假設的母體平均數沒有顯著差異。"

# 範例執行
if __name__ == "__main__":
    # 設定假設的母體平均分數
    population_mean = 80
    
    # 範例一：樣本平均數與母體平均數有顯著差異
    # 假設這個班級的分數較高
    sample_scores_high = [88, 85, 90, 84, 87, 86, 91, 89, 83, 85, 
                          87, 86, 88, 90, 85, 89, 84, 87, 86, 88,
                          92, 85, 87, 89, 86, 88, 85, 90, 87, 86]
    
    t_stat_high, p_value_high = conduct_one_sample_t_test(
        sample_scores_high, 
        population_mean
    )
    result_high = make_decision(p_value_high)

    print("--- 檢定結果 (有顯著差異) ---")
    print(f"假設母體平均數: {population_mean}")
    print(f"樣本平均數: {np.mean(sample_scores_high):.2f}")
    print(f"t 統計量: {t_stat_high:.4f}")
    print(f"p 值: {p_value_high:.4f}")
    print(result_high)

    # 範例二：樣本平均數與母體平均數沒有顯著差異
    # 假設這個班級的分數與學區平均數接近
    sample_scores_no_diff = [81, 78, 82, 79, 80, 83, 77, 80, 79, 81,
                             82, 78, 80, 81, 79, 80, 82, 78, 80, 79,
                             81, 78, 82, 80, 79, 80, 81, 79, 82, 78]

    t_stat_no_diff, p_value_no_diff = conduct_one_sample_t_test(
        sample_scores_no_diff,
        population_mean
    )
    result_no_diff = make_decision(p_value_no_diff)
    
    print("\n--- 檢定結果 (沒有顯著差異) ---")
    print(f"假設母體平均數: {population_mean}")
    print(f"樣本平均數: {np.mean(sample_scores_no_diff):.2f}")
    print(f"t 統計量: {t_stat_no_diff:.4f}")
    print(f"p 值: {p_value_no_diff:.4f}")
    print(result_no_diff)