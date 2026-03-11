mod tensor;
use tensor::Tensor;

fn main() {
    println!("🚀 Mac GPU 神經網路：迎戰經典的 XOR (互斥或) 難題！\n");

    let x = Tensor::new(&[
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ], &[4, 2]);

    let y_true = Tensor::new(&[
        1.0, 0.0,
        0.0, 1.0,
        0.0, 1.0,
        1.0, 0.0,
    ], &[4, 2]);

    println!("🎲 初始化權重 (W) 與 偏差值 (Bias)...");
    
    // 加寬隱藏層 (2 -> 16) 減少 Dead ReLU 造成的影響
    let w1 = Tensor::randn(&[2, 16]);
    let b1 = Tensor::new(&[0.0; 16], &[16]);

    let w2 = Tensor::randn(&[16, 2]);
    let b2 = Tensor::new(&[0.0; 2], &[2]);

    // 降低學習率，增加訓練輪數
    let learning_rate = 0.05; 
    let epochs = 1500;

    for epoch in 1..=epochs {
        w1.zero_grad(); b1.zero_grad();
        w2.zero_grad(); b2.zero_grad();

        // 正向傳播
        let hidden = x.matmul(&w1).add_broadcast(&b1).relu();
        let logits = hidden.matmul(&w2).add_broadcast(&b2);
        let probs = logits.softmax();
        
        let loss = probs.cross_entropy(&y_true);
        loss.backward();

        w1.step(learning_rate); b1.step(learning_rate);
        w2.step(learning_rate); b2.step(learning_rate);

        if epoch % 150 == 0 || epoch == 1 {
            println!("Epoch {:04}/{} | Loss: {:.6}", epoch, epochs, loss.data()[0]);
        }
    }

    println!("\n✅ 訓練完成！看 XOR 問題是否破解：");
    let probs_final = x.matmul(&w1).add_broadcast(&b1).relu().matmul(&w2).add_broadcast(&b2).softmax();
    let preds = probs_final.data();
    
    println!("0 XOR 0 (應為 [1, 0]):[{:.4}, {:.4}]", preds[0], preds[1]);
    println!("0 XOR 1 (應為 [0, 1]): [{:.4}, {:.4}]", preds[2], preds[3]);
    println!("1 XOR 0 (應為 [0, 1]): [{:.4}, {:.4}]", preds[4], preds[5]);
    println!("1 XOR 1 (應為[1, 0]): [{:.4}, {:.4}]", preds[6], preds[7]);
}