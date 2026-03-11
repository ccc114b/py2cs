mod tensor;
use tensor::Tensor;

fn main() {
    println!("🚀 Mac GPU 神經網路：隨機權重初始化 SGD 訓練！\n");

    // 1. 訓練資料 (XOR-like 或是更複雜的二元分類)
    let x = Tensor::new(&[
         1.0,  2.0,
         2.0,  1.0,
        -1.0, -2.0,
        -2.0, -1.0,
         3.0,  3.0,
        -3.0, -3.0,
    ], &[6, 2]);

    // 標籤
    let y_true = Tensor::new(&[
        1.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        0.0, 1.0,
        1.0, 0.0,
        0.0, 1.0,
    ], &[6, 2]);

    // 2. 初始化隨機神經網路權重 (免手打！)
    println!("🎲 正在隨機初始化權重...");
    // 第一層: 2 輸入 -> 8 隱藏節點
    let w1 = Tensor::randn(&[2, 8]);
    // 第二層: 8 隱藏節點 -> 2 輸出類別
    let w2 = Tensor::randn(&[8, 2]);

    let learning_rate = 0.05;
    let epochs = 100; // 這次跑 100 輪

    // 3. 訓練迴圈
    for epoch in 1..=epochs {
        w1.zero_grad();
        w2.zero_grad();

        // 正向傳播
        let hidden = x.matmul(&w1).relu();
        let logits = hidden.matmul(&w2);
        let probs = logits.softmax();
        let loss = probs.cross_entropy(&y_true);

        // 反向傳播
        loss.backward();

        // 參數更新
        w1.step(learning_rate);
        w2.step(learning_rate);

        if epoch % 10 == 0 || epoch == 1 {
            println!("Epoch {:03}/{} | Loss: {:.6}", epoch, epochs, loss.data()[0]);
        }
    }

    // 4. 驗證訓練結果
    println!("\n✅ 訓練完成！看看最後的預測機率：");
    let probs_final = x.matmul(&w1).relu().matmul(&w2).softmax();
    let preds = probs_final.data();
    
    println!("資料 1 (應為[1, 0]): [{:.4}, {:.4}]", preds[0], preds[1]);
    println!("資料 6 (應為 [0, 1]): [{:.4}, {:.4}]", preds[10], preds[11]);
}