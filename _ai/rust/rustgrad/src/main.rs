mod tensor;
mod nn;
mod optimizer;
// 2. 引入模組內的特定結構，方便直接使用
use tensor::Tensor;
use nn::Linear;
use optimizer::SGD;
use ndarray::arr2; // 假設你有用 ndarray

fn main() {
    // 1. 準備訓練資料
    // 輸入 X: 形狀為 (4, 1) 的矩陣
    let x_data = ndarray::arr2(&[[1.0], [2.0], [3.0], [4.0]]).into_dyn();
    // 目標 Y: 形狀為 (4, 1) 的矩陣 (對應 y = 3x + 2)
    let y_data = ndarray::arr2(&[[5.0], [8.0], [11.0], [14.0]]).into_dyn();
    
    let x = Tensor::new(x_data, false);
    let y = Tensor::new(y_data, false);

    // 2. 建立模型與優化器
    let model = Linear::new(1, 1);
    let optimizer = SGD::new(model.parameters(), 0.01);

    // 3. 訓練迴圈
    let epochs = 100;
    for epoch in 0..epochs {
        // --- Forward Pass ---
        let y_pred = model.forward(&x);
        
        // 計算 MSE Loss = sum((y_pred - y)^2) / N 
        // 這裡為了簡化，我們直接用 sum((y_pred - y)^2)
        let diff = &y_pred - &y;
        let loss = diff.pow(2.0).sum();

        // --- Backward Pass ---
        optimizer.zero_grad(); // 1. 清空舊梯度
        loss.backward();       // 2. 反向傳遞計算新梯度
        optimizer.step();      // 3. 更新權重

        // 印出訓練過程
        if epoch % 10 == 0 {
            let current_loss = loss.0.borrow().data.sum(); // 取出純量數值
            let w = model.weight.0.borrow().data[[0, 0]];
            let b = model.bias.0.borrow().data[[0, 0]];
            println!("Epoch {:3} | Loss: {:.4} | W: {:.4}, b: {:.4}", epoch, current_loss, w, b);
        }
    }

    // 觀察最終結果是否接近 W=3.0, b=2.0
    println!("--- Training Complete ---");
    println!("Final Weight: {:.4}", model.weight.0.borrow().data[[0, 0]]);
    println!("Final Bias: {:.4}", model.bias.0.borrow().data[[0, 0]]);
}