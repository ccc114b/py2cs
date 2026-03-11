mod tensor;
use tensor::Tensor;

fn main() {
    println!("🚀 啟動自製 Mac (Metal GPU) 神經網路引擎...");

    // 建立兩層網路的輸入與權重 (這些都會自動進入 GPU 顯示記憶體)
    let x = Tensor::new(&[1.0, 2.0, -1.0], &[1, 3]);
    let w1 = Tensor::new(&[
        0.5, 0.2, -0.1, 
        0.0, 0.3,  0.4, 
       -0.2, 0.1,  0.5
    ], &[3, 3]);
    let w2 = Tensor::new(&[0.1, -0.3, 0.5], &[3, 1]);

    // 目標值 y
    let y_true = Tensor::new(&[1.0], &[1, 1]);

    // --- 正向傳播 (Forward Pass) - 完全在 GPU 執行 ---
    let hidden = x.matmul(&w1).relu();
    let out = hidden.matmul(&w2);
    
    // 計算 MSE-like Loss : sum((out - y_true) * (out - y_true))
    let diff = out.sub(&y_true);
    let loss = diff.mul(&diff).sum();

    println!("\n📊 運算結果 (從 GPU 讀回):");
    println!("預測值 out: {:?}", out.data());
    println!("Loss: {:?}", loss.data());

    // --- 反向傳播 (Backward Pass) - 同樣觸發 GPU 核心運算 ---
    println!("\n🔄 執行反向傳播 (AutoGrad)...");
    loss.backward();

    println!("\n📈 權重的梯度:");
    println!("W2 Grad: {:?}", w2.grad());
    println!("W1 Grad: {:?}", w1.grad());
    println!("\n✅ 完成！");
}