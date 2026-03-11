mod tensor;
mod nn; // 引入我們的新模組

use tensor::Tensor;
use nn::{Linear, SGD};

fn main() {
    println!("🚀 Mac GPU 神經網路：PyTorch 風格 API 升級版！\n");

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

    println!("🏗️  建構神經網路模型與優化器...");
    
    // 建立神經網路層 (我們甚至可以輕鬆擴充成三層、四層)
    let layer1 = Linear::new(2, 16);
    let layer2 = Linear::new(16, 2);

    // 把所有層的參數收集起來，交給 SGD 優化器統一管理
    let mut all_params = Vec::new();
    all_params.extend(layer1.parameters());
    all_params.extend(layer2.parameters());
    
    let optimizer = SGD::new(all_params, 0.05);

    let epochs = 1500;

    // 訓練迴圈變得超級乾淨！
    for epoch in 1..=epochs {
        // 1. 清空梯度
        optimizer.zero_grad();

        // 2. 正向傳播 (Forward)
        let hidden = layer1.forward(&x).relu();
        let logits = layer2.forward(&hidden);
        
        let probs = logits.softmax();
        let loss = probs.cross_entropy(&y_true);

        // 3. 反向傳播 (Backward)
        loss.backward();

        // 4. 參數更新 (Step)
        optimizer.step();

        if epoch % 150 == 0 || epoch == 1 {
            println!("Epoch {:04}/{} | Loss: {:.6}", epoch, epochs, loss.data()[0]);
        }
    }

    println!("\n✅ 訓練完成！看 XOR 問題是否破解：");
    let preds = layer2.forward(&layer1.forward(&x).relu()).softmax().data();
    
    println!("0 XOR 0 (應為[1, 0]):[{:.4}, {:.4}]", preds[0], preds[1]);
    println!("0 XOR 1 (應為 [0, 1]): [{:.4}, {:.4}]", preds[2], preds[3]);
    println!("1 XOR 0 (應為 [0, 1]):[{:.4}, {:.4}]", preds[4], preds[5]);
    println!("1 XOR 1 (應為[1, 0]): [{:.4}, {:.4}]", preds[6], preds[7]);
}