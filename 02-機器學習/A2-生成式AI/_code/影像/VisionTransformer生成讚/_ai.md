
* https://aistudio.google.com/prompts/1yIib7YpSNoXUUWUnlUj9bg5xyysjlw3M
    * 「辨識」是分類問題，而「生成」是回歸問題
    * 在 DiT (生成) 裡，我們是在玩「還原數值」的遊戲，所以用 MSE；
    * 在 ViT (辨識) 裡，我們是在玩「猜標籤」的遊戲，所以用 Cross Entropy。

