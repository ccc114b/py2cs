# AI 語音合成系統

## 執行方式，請參考 test.sh

```sh
python gen_unit.py # 產生單音的音檔（用 gTTS)
python train_unit.py # 訓練單音模型
python test_unit.py # 測試單音模型
python test_direct.py # 用單音模型生成單音後，直接拼接成語句

python gen.py # 產生語句的音檔（用 gTTS)
python train.py # 訓練連接 TTS 模型 (conv1d 只調邊緣版)
python test.py # 測試拼音版模型 (conv1d 只調邊緣版)

# python train_transformer.py # 訓練連接 TTS 模型 (transformer 版)
# python test_transformer.py # 測試拼音版模型 (transformer 版)
```
