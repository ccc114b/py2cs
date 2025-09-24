MAC 上要加下列段落

```py
# 確保你的 Mac 系統有 PingFang.ttc 這個字型，
# 如果沒有，請換成其他你確認可用的中文字型名稱，如 'Arial Unicode MS'
plt.rcParams['font.family'] = 'Arial Unicode MS'

# 解決負號無法正常顯示的問題
plt.rcParams['axes.unicode_minus'] = False
```
