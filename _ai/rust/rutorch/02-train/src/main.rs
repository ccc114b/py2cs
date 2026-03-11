mod tensor;
use tensor::Tensor;

fn main() {
    println!("🚀 Mac GPU 神經網路：開始進行 SGD 訓練！\n");

    // 1. 準備訓練資料 (Dataset)
    // 4 筆資料，每筆 2 個特徵
    let x = Tensor::new(&[
         1.0,  2.0,  // 類別 A
         2.0,  1.0,  // 類別 A
        -1.0, -2.0,  // 類別 B
        -2.0, -1.0,  // 類別 B
    ], &[4, 2]);

    // 4 筆資料對應的 One-Hot 標籤 (2個類別)
    let y_true = Tensor::new(&[
        1.0, 0.0,    // 類別 A
        1.0, 0.0,    // 類別 A
        0.0, 1.0,    // 類別 B
        0.0, 1.0,    // 類別 B
    ], &[4, 2]);

    // 2. 初始化神經網路權重 (Weights)
    // 第一層: 2 輸入 -> 4 隱藏節點
    let w1 = Tensor::new(&[
        0.1, -0.2, 0.3, -0.1,
        0.5,  0.1, -0.4, 0.2
    ], &[2, 4]);

    // 第二層: 4 隱藏節點 -> 2 輸出類別
    let w2 = Tensor::new(&[
        0.2, -0.1,
       -0.3,  0.4,
        0.1,  0.1,
       -0.2,  0.5
    ], &[4, 2]);

    let learning_rate = 0.05;
    let epochs = 50; // 訓練 50 輪

    // 3. 訓練迴圈 (Training Loop)
    for epoch in 1..=epochs {
        // --- 步驟 A：清空梯度 ---
        w1.zero_grad();
        w2.zero_grad();

        // --- 步驟 B：正向傳播 (Forward Pass) ---
        let hidden = x.matmul(&w1).relu();
        let logits = hidden.matmul(&w2);
        let probs = logits.softmax();
        let loss = probs.cross_entropy(&y_true);

        // --- 步驟 C：反向傳播 (Backward Pass) ---
        loss.backward();

        // --- 步驟 D：更新權重 (Gradient Descent Step) ---
        w1.step(learning_rate);
        w2.step(learning_rate);

        // 每 10 輪印出一次當前進度
        if epoch % 10 == 0 || epoch == 1 {
            let current_loss = loss.data()[0];
            println!("Epoch {:02}/{} | Loss: {:.6}", epoch, epochs, current_loss);
        }
    }

    // 4. 驗證訓練結果
    println!("\n✅ 訓練完成！來看看模型最後的預測機率：");
    let hidden_final = x.matmul(&w1).relu();
    let logits_final = hidden_final.matmul(&w2);
    let probs_final = logits_final.softmax();
    
    let preds = probs_final.data();
    println!("第一筆資料 (應為 [1, 0]):[{:.4}, {:.4}]", preds[0], preds[1]);
    println!("第二筆資料 (應為[1, 0]): [{:.4}, {:.4}]", preds[2], preds[3]);
    println!("第三筆資料 (應為 [0, 1]): [{:.4}, {:.4}]", preds[4], preds[5]);
    println!("第四筆資料 (應為 [0, 1]):[{:.4}, {:.4}]", preds[6], preds[7]);
}