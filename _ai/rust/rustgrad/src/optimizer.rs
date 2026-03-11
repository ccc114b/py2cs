use crate::tensor::Tensor;

pub struct SGD {
    params: Vec<Tensor>,
    lr: f64,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        SGD { params, lr }
    }

    /// 將所有參數的梯度歸零
    pub fn zero_grad(&self) {
        for p in &self.params {
            let mut inner = p.0.borrow_mut();
            inner.grad.fill(0.0);
        }
    }

    /// 執行一步參數更新： data = data - lr * grad
    pub fn step(&self) {
        for p in &self.params {
            let mut inner = p.0.borrow_mut();
            // 由於內部儲存的是 ndarray，可以直接做陣列運算
            let grad_step = &inner.grad * self.lr;
            inner.data = &inner.data - &grad_step;
        }
    }
}