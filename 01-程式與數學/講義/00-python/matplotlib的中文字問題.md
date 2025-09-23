加入下列程式碼

```py
# 設定 Matplotlib 支援中文的字型
# 'SimHei' 是 Windows 系統常見的中文黑體字型
# 'Microsoft JhengHei' 適用於繁體中文
# 'WenQuanYi Zen Hei' 適用於 Linux
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # Windows
plt.rcParams['font.sans-serif'] = ['Heiti TC'] # macOS
#plt.rcParams['font.family'] = 'Arial Unicode MS'
# 解決負號'-'顯示為方塊的問題
plt.rcParams['axes.unicode_minus'] = False 
```
