use ndarray::ArrayD;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Mul, Sub}; // 確保導入 Sub
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Op {
    None,
    Add,
    Mul,
    Pow(f64),
    Relu,
    Matmul,
    Sub, 
    Sum,
}

pub struct TensorData {
    pub data: ArrayD<f64>,
    pub grad: ArrayD<f64>,
    pub requires_grad: bool,
    pub prev: Vec<Tensor>,
    pub op: Op,
}

#[derive(Clone)]
// 修正 E0616: 將第一個欄位設為 pub，這樣 main.rs 和 optimizer.rs 才能用 .0
pub struct Tensor(pub Rc<RefCell<TensorData>>);

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Tensor {}

impl Tensor {
    pub fn new(data: ArrayD<f64>, requires_grad: bool) -> Self {
        let grad = ArrayD::zeros(data.dim());
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad,
            requires_grad,
            prev: vec![],
            op: Op::None,
        })))
    }

    fn _new_node(data: ArrayD<f64>, prev: Vec<Tensor>, op: Op) -> Self {
        let grad = ArrayD::zeros(data.dim());
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad,
            requires_grad: true,
            prev,
            op,
        })))
    }

    pub fn relu(&self) -> Self {
        let data = self.0.borrow().data.mapv(|x| if x > 0.0 { x } else { 0.0 });
        Self::_new_node(data, vec![self.clone()], Op::Relu)
    }

    pub fn pow(&self, exp: f64) -> Self {
        let data = self.0.borrow().data.mapv(|x| x.powf(exp));
        Self::_new_node(data, vec![self.clone()], Op::Pow(exp))
    }

    pub fn matmul(&self, other: &Tensor) -> Self {
        let a = self.0.borrow().data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b = other.0.borrow().data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let data = a.dot(&b).into_dyn();
        Self::_new_node(data, vec![self.clone(), other.clone()], Op::Matmul)
    }

    pub fn sum(&self) -> Self {
        let sum_val = self.0.borrow().data.sum();
        let data = ndarray::arr0(sum_val).into_dyn();
        Self::_new_node(data, vec![self.clone()], Op::Sum)
    }

    fn backward_step(&self) {
        let mut inner = self.0.borrow_mut();
        let grad = inner.grad.clone();
        
        // 修正 E0004: 補齊所有 Op 的梯度計算邏輯
        match &inner.op {
            Op::Add => {
                let mut prev0 = inner.prev[0].0.borrow_mut();
                let mut prev1 = inner.prev[1].0.borrow_mut();
                prev0.grad = &prev0.grad + &grad;
                prev1.grad = &prev1.grad + &grad;
            }
            Op::Sub => {
                let mut prev0 = inner.prev[0].0.borrow_mut();
                let mut prev1 = inner.prev[1].0.borrow_mut();
                prev0.grad = &prev0.grad + &grad;
                prev1.grad = &prev1.grad - &grad;
            }
            Op::Mul => {
                let mut prev0 = inner.prev[0].0.borrow_mut();
                let mut prev1 = inner.prev[1].0.borrow_mut();
                let data0 = prev0.data.clone();
                let data1 = prev1.data.clone();
                prev0.grad = &prev0.grad + &(&data1 * &grad);
                prev1.grad = &prev1.grad + &(&data0 * &grad);
            }
            Op::Relu => {
                let mut prev0 = inner.prev[0].0.borrow_mut();
                let out_data = &inner.data;
                let relu_grad = out_data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                prev0.grad = &prev0.grad + &(&relu_grad * &grad);
            }
            Op::Pow(exp) => {
                let mut prev0 = inner.prev[0].0.borrow_mut();
                let data0 = prev0.data.clone();
                let pow_grad = data0.mapv(|x| exp * x.powf(exp - 1.0));
                prev0.grad = &prev0.grad + &(&pow_grad * &grad);
            }
            Op::Matmul => {
                let mut prev0 = inner.prev[0].0.borrow_mut();
                let mut prev1 = inner.prev[1].0.borrow_mut();
                let a = prev0.data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                let b = prev1.data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                let g = grad.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                let grad_a = g.dot(&b.t());
                let grad_b = a.t().dot(&g);
                prev0.grad = &prev0.grad + &grad_a.into_dyn();
                prev1.grad = &prev1.grad + &grad_b.into_dyn();
            }
            Op::Sum => {
                let mut prev0 = inner.prev[0].0.borrow_mut();
                let g_val = grad.sum(); // Sum 的結果是純量，將此值廣播到所有輸入梯度
                prev0.grad.mapv_inplace(|x| x + g_val);
            }
            Op::None => {} 
        }
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(v: &Tensor, topo: &mut Vec<Tensor>, visited: &mut HashSet<Tensor>) {
            if !visited.contains(v) {
                visited.insert(v.clone());
                for child in &v.0.borrow().prev {
                    build_topo(child, topo, visited);
                }
                topo.push(v.clone());
            }
        }

        build_topo(self, &mut topo, &mut visited);
        self.0.borrow_mut().grad.fill(1.0);

        for v in topo.iter().rev() {
            v.backward_step();
        }
    }
}

// 運算子多載
impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        let data = &self.0.borrow().data + &rhs.0.borrow().data;
        Tensor::_new_node(data, vec![self.clone(), rhs.clone()], Op::Add)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        let data = &self.0.borrow().data - &rhs.0.borrow().data;
        Tensor::_new_node(data, vec![self.clone(), rhs.clone()], Op::Sub)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        let data = &self.0.borrow().data * &rhs.0.borrow().data;
        Tensor::_new_node(data, vec![self.clone(), rhs.clone()], Op::Mul)
    }
}