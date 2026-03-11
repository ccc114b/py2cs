use ndarray::ArrayD;
use crate::tensor::Tensor;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    /// 初始化 Linear 層
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // 為了簡化範例，這裡使用固定小數值初始化。實務上應使用常態分佈隨機初始化。
        let w = ArrayD::from_elem(vec![in_features, out_features], 0.1);
        let b = ArrayD::from_elem(vec![1, out_features], 0.0);
        
        Linear {
            weight: Tensor::new(w, true),
            bias: Tensor::new(b, true),
        }
    }

    /// 前向傳遞： X * W + b
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let xw = x.matmul(&self.weight);
        &xw + &self.bias
    }

    /// 回傳此層所有需要優化的參數
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}