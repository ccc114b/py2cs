import numpy as np
from scipy import stats

def conduct_t_test(new_app_scores, traditional_scores):
    """
    對兩組學生的數學成績進行獨立樣本 t 檢定。

    Args:
        new_app_scores (list or numpy.array): 使用新 App 的學生分數。
        traditional_scores (list or numpy.array): 使用傳統教學方法的學生分數。

    Returns:
        tuple: 包含 t 統計量和 p 值。
    """
    # 確保兩組樣本數都大於 1
    if len(new_app_scores) < 2 or len(traditional_scores) < 2:
        raise ValueError("兩組樣本數都必須大於 1")
    
    # 使用獨立樣本 t 檢定 (假設變異數不相等，使用 Welch's t-test 更穩健)
    t_stat, p_value = stats.ttest_ind(
        new_app_scores, 
        traditional_scores, 
        equal_var=False, 
        alternative='greater' # 這裡指定對立假設為 'greater' (單尾檢定)
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
        return f"p 值 ({p_value:.4f}) 小於顯著水準 ({alpha})，有足夠證據拒絕虛無假設。\n結論：新 App 顯著提高了學生成績。"
    else:
        return f"p 值 ({p_value:.4f}) 大於顯著水準 ({alpha})，沒有足夠證據拒絕虛無假設。\n結論：新 App 沒有顯著效果。"

# 範例執行
if __name__ == "__main__":
    # 直接寫入數據：新 App 組的成績分數顯著較高 (資料數量：30 個)
    new_app_scores = [
        86, 88, 85, 92, 89, 87, 84, 91, 93, 80,
        90, 85, 88, 91, 86, 94, 87, 89, 92, 85,
        88, 86, 90, 85, 87, 91, 93, 86, 88, 84
    ]
    traditional_scores = [
        75, 78, 72, 80, 76, 74, 77, 79, 73, 81,
        75, 78, 76, 74, 79, 72, 80, 75, 77, 76,
        78, 74, 73, 79, 75, 76, 78, 77, 75, 72
    ]
    
    t_stat, p_value = conduct_t_test(new_app_scores, traditional_scores)
    result_text = make_decision(p_value)

    print("--- 檢定結果 (新 App 成績顯著較高) ---")
    print(f"t 統計量: {t_stat:.4f}")
    print(f"p 值: {p_value:.4f}")
    print(result_text)

    # 直接寫入另一組數據：新 App 組的成績分數沒有顯著差異 (資料數量：30 個)
    new_app_scores_no_diff = [
        80, 82, 78, 85, 81, 79, 83, 84, 80, 77,
        82, 80, 79, 81, 83, 78, 84, 81, 80, 82,
        79, 81, 83, 80, 79, 84, 82, 81, 78, 80
    ]
    traditional_scores_no_diff = [
        78, 81, 80, 83, 79, 80, 82, 81, 77, 80,
        81, 78, 82, 79, 80, 83, 79, 80, 82, 81,
        77, 80, 82, 79, 81, 80, 83, 81, 78, 80
    ]
    
    t_stat_no_diff, p_value_no_diff = conduct_t_test(
        new_app_scores_no_diff, 
        traditional_scores_no_diff
    )
    result_text_no_diff = make_decision(p_value_no_diff)
    
    print("\n--- 檢定結果 (新 App 成績沒有顯著差異) ---")
    print(f"t 統計量: {t_stat_no_diff:.4f}")
    print(f"p 值: {p_value_no_diff:.4f}")
    print(result_text_no_diff)