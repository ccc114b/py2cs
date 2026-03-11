#![allow(dead_code)]
use lazy_static::lazy_static;
use metal::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ptr;
use std::rc::Rc;
// 新增這兩行來引入常態分佈亂數
use rand::thread_rng;
use rand_distr::{Normal, Distribution};

// ==========================================
// 1. Metal GPU 環境與著色器 (MSL) 程式碼
// ==========================================

const SHADER_CODE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}
kernel void mul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * b[id];
}
kernel void add_assign(device float* target [[buffer(0)]], device const float* source [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    target[id] += source[id];
}
kernel void relu(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    out[id] = max(0.0, a[id]);
}
kernel void relu_backward(device const float* a [[buffer(0)]], device const float* grad_in [[buffer(1)]], device float* grad_out [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    grad_out[id] = a[id] > 0.0 ? grad_in[id] : 0.0;
}
kernel void matmul(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], uint2 id [[thread_position_in_grid]]) {
    if (id.x >= N || id.y >= M) return;
    float sum = 0.0;
    for (uint i = 0; i < K; ++i) { sum += A[id.y * K + i] * B[i * N + id.x]; }
    C[id.y * N + id.x] = sum;
}
kernel void transpose(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint2 id [[thread_position_in_grid]]) {
    if (id.x >= cols || id.y >= rows) return;
    out[id.x * rows + id.y] = in[id.y * cols + id.x];
}
kernel void sum_all(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& length [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id > 0) return;
    float s = 0.0;
    for (uint i = 0; i < length; ++i) { s += in[i]; }
    out[0] = s;
}
kernel void sum_backward(device float* target_grad [[buffer(0)]], device const float* out_grad [[buffer(1)]], constant uint& length [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id >= length) return;
    target_grad[id] += out_grad[0];
}

// === 新增：Power (次方) ===
kernel void power(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], constant float& p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    out[id] = pow(a[id], p);
}
kernel void power_backward(device const float* a [[buffer(0)]], device const float* grad_out [[buffer(1)]], device float* target_grad [[buffer(2)]], constant float& p [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    target_grad[id] += p * pow(a[id], p - 1.0) * grad_out[id];
}

// === 新增：Log (自然對數) ===
kernel void log_fw(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    out[id] = log(a[id]);
}
kernel void log_bw(device const float* a [[buffer(0)]], device const float* grad_out [[buffer(1)]], device float* target_grad [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    target_grad[id] += (1.0 / a[id]) * grad_out[id];
}

// === 新增：Softmax (2D, 針對列做正規化) ===
// 這裡 id 代表的是第幾列 (row)
kernel void softmax_fw(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& cols [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    uint offset = id * cols;
    // 尋找最大值以確保數值穩定性
    float max_val = in[offset];
    for (uint i = 1; i < cols; ++i) { max_val = max(max_val, in[offset + i]); }
    
    float sum = 0.0;
    for (uint i = 0; i < cols; ++i) {
        float e = exp(in[offset + i] - max_val);
        out[offset + i] = e;
        sum += e;
    }
    for (uint i = 0; i < cols; ++i) { out[offset + i] /= sum; }
}
kernel void softmax_bw(device const float* sm_out [[buffer(0)]], device const float* grad_out [[buffer(1)]], device float* target_grad [[buffer(2)]], constant uint& cols [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    uint offset = id * cols;
    float s = 0.0;
    for (uint i = 0; i < cols; ++i) { s += grad_out[offset + i] * sm_out[offset + i]; }
    for (uint i = 0; i < cols; ++i) { target_grad[offset + i] += (grad_out[offset + i] - s) * sm_out[offset + i]; }
}

// === 新增：優化器與梯度管理 ===

// 清空梯度 (設為 0)
kernel void zero_grad(device float* grad [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    grad[id] = 0.0;
}

// SGD 權重更新： param = param - lr * grad
kernel void sgd_step(device float* param [[buffer(0)]], device const float* grad [[buffer(1)]], constant float& lr [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    param[id] -= lr * grad[id];
}

// === 新增：廣播加法 (Bias 加法) ===
// 將一維的 b (長度為 cols) 廣播加到二維的 a (大小為 rows x cols) 上
kernel void add_broadcast(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& cols [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    // id % cols 可以精準地讓每一列對應到正確的 b 元素
    out[id] = a[id] + b[id % cols];
}

// 廣播加法的反向傳播 (專門計算 Bias 的梯度)
// Bias 的梯度等於「所有批次(Batch)的梯度在垂直方向的加總」
kernel void add_broadcast_bw_b(device const float* grad_out [[buffer(0)]], device float* grad_b [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    if (id >= cols) return;
    float sum = 0.0;
    // 將每個 column (對應一個 bias) 的梯度往下加總
    for(uint r = 0; r < rows; r++) {
        sum += grad_out[r * cols + id];
    }
    grad_b[id] += sum;
}
"#;

struct MetalContext { device: Device, queue: CommandQueue, library: Library }
impl MetalContext {
    fn new() -> Self {
        let device = Device::system_default().expect("找不到 Metal 設備！");
        let queue = device.new_command_queue();
        let options = CompileOptions::new();
        let library = device.new_library_with_source(SHADER_CODE, &options).unwrap();
        Self { device, queue, library }
    }
}
lazy_static! { static ref METAL_CTX: MetalContext = MetalContext::new(); }

#[derive(Clone)]
pub struct GpuBuffer { pub buffer: Buffer, pub length: usize }
impl GpuBuffer {
    fn new(data: &[f32]) -> Self {
        let size = (data.len() * std::mem::size_of::<f32>()) as u64;
        let buffer = METAL_CTX.device.new_buffer_with_data(data.as_ptr() as *const _, size, MTLResourceOptions::StorageModeShared);
        Self { buffer, length: data.len() }
    }
    fn zeros(length: usize) -> Self { Self::new(&vec![0.0f32; length]) }
    pub fn to_vec(&self) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.length];
        let ptr = self.buffer.contents() as *const f32;
        unsafe { std::ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), self.length); }
        vec
    }
}

// ==========================================
// 2. GPU 指令分派輔助函式 (Dispatchers)
// ==========================================

fn dispatch_1d(name: &str, buffers: &[&Buffer], length: usize) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function(name, None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for (i, buf) in buffers.iter().enumerate() { encoder.set_buffer(i as u64, Some(*buf), 0); }
    let grid_size = MTLSize::new(length as u64, 1, 1);
    let tg_size = MTLSize::new(std::cmp::min(length as u64, 32).max(1), 1, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit(); cmd_buffer.wait_until_completed();
}

// 支援傳遞 u32 變數 (例如 cols 或 length)
fn dispatch_1d_with_u32(name: &str, buffers: &[&Buffer], length: usize, val: u32) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function(name, None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for (i, buf) in buffers.iter().enumerate() { encoder.set_buffer(i as u64, Some(*buf), 0); }
    encoder.set_bytes(buffers.len() as u64, 4, &val as *const _ as *const _); // 綁定最後一個參數
    let grid_size = MTLSize::new(length as u64, 1, 1);
    let tg_size = MTLSize::new(std::cmp::min(length as u64, 32).max(1), 1, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit(); cmd_buffer.wait_until_completed();
}

// 支援傳遞兩個 u32 變數 (rows, cols)
fn dispatch_1d_with_2_u32(name: &str, buffers: &[&Buffer], length: usize, val1: u32, val2: u32) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function(name, None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    
    for (i, buf) in buffers.iter().enumerate() { encoder.set_buffer(i as u64, Some(*buf), 0); }
    encoder.set_bytes(buffers.len() as u64, 4, &val1 as *const _ as *const _);
    encoder.set_bytes((buffers.len() + 1) as u64, 4, &val2 as *const _ as *const _);
    
    let grid_size = MTLSize::new(length as u64, 1, 1);
    let tg_size = MTLSize::new(std::cmp::min(length as u64, 32).max(1), 1, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit(); cmd_buffer.wait_until_completed();
}

// 支援傳遞 f32 變數 (例如 power)
fn dispatch_1d_with_f32(name: &str, buffers: &[&Buffer], length: usize, val: f32) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function(name, None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for (i, buf) in buffers.iter().enumerate() { encoder.set_buffer(i as u64, Some(*buf), 0); }
    encoder.set_bytes(buffers.len() as u64, 4, &val as *const _ as *const _);
    let grid_size = MTLSize::new(length as u64, 1, 1);
    let tg_size = MTLSize::new(std::cmp::min(length as u64, 32).max(1), 1, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit(); cmd_buffer.wait_until_completed();
}

fn dispatch_matmul(name: &str, a: &Buffer, b: &Buffer, c: &Buffer, m: u32, k: u32, n: u32) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function(name, None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(a), 0); encoder.set_buffer(1, Some(b), 0); encoder.set_buffer(2, Some(c), 0);
    encoder.set_bytes(3, 4, &m as *const _ as *const _); encoder.set_bytes(4, 4, &k as *const _ as *const _); encoder.set_bytes(5, 4, &n as *const _ as *const _);
    let grid_size = MTLSize::new(n as u64, m as u64, 1);
    let tg_size = MTLSize::new(8, 8, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit(); cmd_buffer.wait_until_completed();
}

fn dispatch_transpose(in_buf: &Buffer, out_buf: &Buffer, rows: u32, cols: u32) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function("transpose", None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(in_buf), 0); encoder.set_buffer(1, Some(out_buf), 0);
    encoder.set_bytes(2, 4, &rows as *const _ as *const _); encoder.set_bytes(3, 4, &cols as *const _ as *const _);
    let grid_size = MTLSize::new(cols as u64, rows as u64, 1);
    let tg_size = MTLSize::new(8, 8, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit(); cmd_buffer.wait_until_completed();
}

// ==========================================
// 3. 張量引擎與反向傳播 (Autograd)
// ==========================================

pub enum Op {
    Leaf,
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Matmul(Tensor, Tensor),
    Relu(Tensor),
    Sum(Tensor),
    Pow(Tensor, f32),   // 次方
    Log(Tensor),        // 對數
    Softmax(Tensor),    // Softmax
    AddBroadcast(Tensor, Tensor), // 廣播加法
}

pub struct TensorInner {
    pub data: GpuBuffer,
    pub grad: GpuBuffer,
    pub shape: Vec<usize>,
    pub op: Op,
}

#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<TensorInner>>);

impl PartialEq for Tensor { fn eq(&self, other: &Self) -> bool { Rc::ptr_eq(&self.0, &other.0) } }
impl Eq for Tensor {}
impl Hash for Tensor { fn hash<H: Hasher>(&self, state: &mut H) { ptr::hash(Rc::as_ptr(&self.0), state) } }

impl Tensor {
    pub fn new(data: &[f32], shape: &[usize]) -> Self {
        let _ = &*METAL_CTX; 
        Self(Rc::new(RefCell::new(TensorInner { data: GpuBuffer::new(data), grad: GpuBuffer::zeros(data.len()), shape: shape.to_vec(), op: Op::Leaf })))
    }

    pub fn data(&self) -> Vec<f32> { self.0.borrow().data.to_vec() }
    pub fn grad(&self) -> Vec<f32> { self.0.borrow().grad.to_vec() }
    pub fn shape(&self) -> Vec<usize> { self.0.borrow().shape.clone() }

    /// 產生標準常態分佈 (Mean=0, Std=1) 的隨機張量，類似 PyTorch 的 torch.randn
    pub fn randn(shape: &[usize]) -> Self {
        let mut rng = thread_rng();
        // 為了讓神經網路好訓練，我們使用標準差 0.1 (類似簡單的權重初始化)
        let normal = Normal::new(0.0, 0.1).unwrap(); 
        
        let length: usize = shape.iter().product();
        let mut data = Vec::with_capacity(length);
        for _ in 0..length {
            data.push(normal.sample(&mut rng) as f32);
        }
        
        Self::new(&data, shape)
    }
    
    pub fn add(&self, other: &Tensor) -> Tensor {
        let len = self.0.borrow().data.length;
        let out_buf = GpuBuffer::zeros(len);
        dispatch_1d("add", &[&self.0.borrow().data.buffer, &other.0.borrow().data.buffer, &out_buf.buffer], len);
        Self(Rc::new(RefCell::new(TensorInner { data: out_buf, grad: GpuBuffer::zeros(len), shape: self.shape(), op: Op::Add(self.clone(), other.clone()) })))
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let len = self.0.borrow().data.length;
        let out_buf = GpuBuffer::zeros(len);
        dispatch_1d("mul", &[&self.0.borrow().data.buffer, &other.0.borrow().data.buffer, &out_buf.buffer], len);
        Self(Rc::new(RefCell::new(TensorInner { data: out_buf, grad: GpuBuffer::zeros(len), shape: self.shape(), op: Op::Mul(self.clone(), other.clone()) })))
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let m = self.shape()[0] as u32; let k = self.shape()[1] as u32; let n = other.shape()[1] as u32;
        let out_buf = GpuBuffer::zeros((m * n) as usize);
        dispatch_matmul("matmul", &self.0.borrow().data.buffer, &other.0.borrow().data.buffer, &out_buf.buffer, m, k, n);
        Self(Rc::new(RefCell::new(TensorInner { data: out_buf, grad: GpuBuffer::zeros((m * n) as usize), shape: vec![m as usize, n as usize], op: Op::Matmul(self.clone(), other.clone()) })))
    }

    pub fn relu(&self) -> Tensor {
        let len = self.0.borrow().data.length;
        let out_buf = GpuBuffer::zeros(len);
        dispatch_1d("relu", &[&self.0.borrow().data.buffer, &out_buf.buffer], len);
        Self(Rc::new(RefCell::new(TensorInner { data: out_buf, grad: GpuBuffer::zeros(len), shape: self.shape(), op: Op::Relu(self.clone()) })))
    }

    pub fn sum(&self) -> Tensor {
        let len = self.0.borrow().data.length;
        let out_buf = GpuBuffer::zeros(1);
        dispatch_1d_with_u32("sum_all", &[&self.0.borrow().data.buffer, &out_buf.buffer], len, len as u32);
        Self(Rc::new(RefCell::new(TensorInner { data: out_buf, grad: GpuBuffer::zeros(1), shape: vec![1, 1], op: Op::Sum(self.clone()) })))
    }

    // --- 新增的函數 ---
    
    pub fn pow(&self, p: f32) -> Tensor {
        let len = self.0.borrow().data.length;
        let out_buf = GpuBuffer::zeros(len);
        dispatch_1d_with_f32("power", &[&self.0.borrow().data.buffer, &out_buf.buffer], len, p);
        Self(Rc::new(RefCell::new(TensorInner { data: out_buf, grad: GpuBuffer::zeros(len), shape: self.shape(), op: Op::Pow(self.clone(), p) })))
    }

    pub fn log(&self) -> Tensor {
        let len = self.0.borrow().data.length;
        let out_buf = GpuBuffer::zeros(len);
        dispatch_1d("log_fw", &[&self.0.borrow().data.buffer, &out_buf.buffer], len);
        Self(Rc::new(RefCell::new(TensorInner { data: out_buf, grad: GpuBuffer::zeros(len), shape: self.shape(), op: Op::Log(self.clone()) })))
    }

    pub fn softmax(&self) -> Tensor {
        let shape = self.shape();
        let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, shape[0]) };
        let out_buf = GpuBuffer::zeros(rows * cols);
        // Dispatcher grid size = rows，因為一個 Thread 負責一整列的 Softmax
        dispatch_1d_with_u32("softmax_fw", &[&self.0.borrow().data.buffer, &out_buf.buffer], rows, cols as u32);
        Self(Rc::new(RefCell::new(TensorInner { data: out_buf, grad: GpuBuffer::zeros(rows * cols), shape: self.shape(), op: Op::Softmax(self.clone()) })))
    }

    pub fn neg(&self) -> Tensor {
        let len = self.0.borrow().data.length;
        let neg_ones = Tensor::new(&vec![-1.0; len], &self.shape());
        self.mul(&neg_ones)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.add(&other.neg())
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        self.mul(&other.pow(-1.0))
    }

    pub fn cross_entropy(&self, yb: &Tensor) -> Tensor {
        let log_probs = self.log();
        let zb = yb.mul(&log_probs);
        zb.sum().neg() // 在數學上 global_sum == sum(axis=1).sum()
    }

    pub fn add_broadcast(&self, b: &Tensor) -> Tensor {
        let len = self.0.borrow().data.length;
        let cols = b.0.borrow().data.length as u32;
        let out_buf = GpuBuffer::zeros(len);
        
        dispatch_1d_with_u32("add_broadcast", &[&self.0.borrow().data.buffer, &b.0.borrow().data.buffer, &out_buf.buffer], len, cols);
        
        Self(Rc::new(RefCell::new(TensorInner {
            data: out_buf, grad: GpuBuffer::zeros(len), shape: self.shape(), op: Op::AddBroadcast(self.clone(), b.clone())
        })))
    }
    
    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(v: &Tensor, visited: &mut HashSet<Tensor>, topo: &mut Vec<Tensor>) {
            if !visited.contains(v) {
                visited.insert(v.clone());
                match &v.0.borrow().op {
                    Op::Add(a, b) | Op::Mul(a, b) | Op::Matmul(a, b) | Op::AddBroadcast(a, b) => { build_topo(a, visited, topo); build_topo(b, visited, topo); }
                    Op::Relu(a) | Op::Sum(a) | Op::Pow(a, _) | Op::Log(a) | Op::Softmax(a) => { build_topo(a, visited, topo); }
                    Op::Leaf => {}
                }
                topo.push(v.clone());
            }
        }
        build_topo(self, &mut visited, &mut topo);

        let len = self.0.borrow().data.length;
        self.0.borrow_mut().grad = GpuBuffer::new(&vec![1.0; len]);

        for node in topo.into_iter().rev() {
            let inner = node.0.borrow();
            let grad = &inner.grad;

            match &inner.op {
                Op::Add(a, b) => {
                    dispatch_1d("add_assign", &[&a.0.borrow().grad.buffer, &grad.buffer], grad.length);
                    dispatch_1d("add_assign", &[&b.0.borrow().grad.buffer, &grad.buffer], grad.length);
                }
                Op::Mul(a, b) => {
                    let tmp_a = GpuBuffer::zeros(grad.length);
                    dispatch_1d("mul", &[&b.0.borrow().data.buffer, &grad.buffer, &tmp_a.buffer], grad.length);
                    dispatch_1d("add_assign", &[&a.0.borrow().grad.buffer, &tmp_a.buffer], grad.length);

                    let tmp_b = GpuBuffer::zeros(grad.length);
                    dispatch_1d("mul", &[&a.0.borrow().data.buffer, &grad.buffer, &tmp_b.buffer], grad.length);
                    dispatch_1d("add_assign", &[&b.0.borrow().grad.buffer, &tmp_b.buffer], grad.length);
                }
                Op::Matmul(a, b) => {
                    let m = a.shape()[0] as u32; let k = a.shape()[1] as u32; let n = b.shape()[1] as u32;
                    let b_t = GpuBuffer::zeros((n * k) as usize); dispatch_transpose(&b.0.borrow().data.buffer, &b_t.buffer, k, n);
                    let a_grad_update = GpuBuffer::zeros((m * k) as usize); dispatch_matmul("matmul", &grad.buffer, &b_t.buffer, &a_grad_update.buffer, m, n, k);
                    dispatch_1d("add_assign", &[&a.0.borrow().grad.buffer, &a_grad_update.buffer], (m * k) as usize);

                    let a_t = GpuBuffer::zeros((k * m) as usize); dispatch_transpose(&a.0.borrow().data.buffer, &a_t.buffer, m, k);
                    let b_grad_update = GpuBuffer::zeros((k * n) as usize); dispatch_matmul("matmul", &a_t.buffer, &grad.buffer, &b_grad_update.buffer, k, m, n);
                    dispatch_1d("add_assign", &[&b.0.borrow().grad.buffer, &b_grad_update.buffer], (k * n) as usize);
                }
                Op::Relu(a) => {
                    let tmp = GpuBuffer::zeros(grad.length);
                    dispatch_1d("relu_backward", &[&a.0.borrow().data.buffer, &grad.buffer, &tmp.buffer], grad.length);
                    dispatch_1d("add_assign", &[&a.0.borrow().grad.buffer, &tmp.buffer], grad.length);
                }
                Op::Sum(a) => {
                    dispatch_1d_with_u32("sum_backward", &[&a.0.borrow().grad.buffer, &grad.buffer], a.0.borrow().grad.length, a.0.borrow().grad.length as u32);
                }
                Op::Pow(a, p) => {
                    let len = grad.length;
                    dispatch_1d_with_f32("power_backward", &[&a.0.borrow().data.buffer, &grad.buffer, &a.0.borrow().grad.buffer], len, *p);
                }
                Op::Log(a) => {
                    let len = grad.length;
                    dispatch_1d("log_bw", &[&a.0.borrow().data.buffer, &grad.buffer, &a.0.borrow().grad.buffer], len);
                }
                Op::Softmax(a) => {
                    let shape = a.shape();
                    let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, shape[0]) };
                    // softmax_bw 需要 softmax 的結果(inner.data)、外層梯度(grad)，並累加到原本節點的梯度(a.grad)
                    dispatch_1d_with_u32("softmax_bw", &[&inner.data.buffer, &grad.buffer, &a.0.borrow().grad.buffer], rows, cols as u32);
                }
                Op::AddBroadcast(a, b) => {
                    // dLoss/da 就是原本的梯度 (因為 a+b 的微分是 1)
                    dispatch_1d("add_assign", &[&a.0.borrow().grad.buffer, &grad.buffer], grad.length);

                    // dLoss/db 需要把梯度往下加總
                    let cols = b.0.borrow().data.length as u32;
                    let rows = (grad.length as u32) / cols;
                    // 指派 thread 數量等於 cols，每個 thread 負責加總一個 column
                    dispatch_1d_with_2_u32("add_broadcast_bw_b", &[&grad.buffer, &b.0.borrow().grad.buffer], cols as usize, rows, cols);
                }
                Op::Leaf => {}
            }
        }
    }

    // --- 優化器方法 (Optimizer) ---

    /// 清空這個張量的梯度 (在每個 Batch 訓練開始前呼叫)
    pub fn zero_grad(&self) {
        let len = self.0.borrow().grad.length;
        // 呼叫 GPU 直接把記憶體歸零
        dispatch_1d("zero_grad", &[&self.0.borrow().grad.buffer], len);
    }

    /// SGD 梯度下降更新：W = W - lr * W.grad
    pub fn step(&self, lr: f32) {
        let len = self.0.borrow().data.length;
        // 讓 GPU 平行執行參數更新
        dispatch_1d_with_f32("sgd_step", &[
            &self.0.borrow().data.buffer, 
            &self.0.borrow().grad.buffer
        ], len, lr);
    }
}