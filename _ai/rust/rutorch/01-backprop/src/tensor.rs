use lazy_static::lazy_static;
use metal::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ptr;
use std::rc::Rc;

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

// 注意 buffer(2) 是 length
kernel void sum_all(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& length [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id > 0) return;
    float s = 0.0;
    for (uint i = 0; i < length; ++i) { s += in[i]; }
    out[0] = s;
}

// 注意 buffer(2) 是 length
kernel void sum_backward(device float* target_grad [[buffer(0)]], device const float* out_grad [[buffer(1)]], constant uint& length [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id >= length) return;
    target_grad[id] += out_grad[0];
}
"#;

struct MetalContext {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl MetalContext {
    fn new() -> Self {
        let device = Device::system_default().expect("找不到 Metal 設備，請確認你在 Mac 上執行！");
        let queue = device.new_command_queue();
        let options = CompileOptions::new();
        let library = device.new_library_with_source(SHADER_CODE, &options).unwrap();
        Self { device, queue, library }
    }
}

lazy_static! {
    static ref METAL_CTX: MetalContext = MetalContext::new();
}

#[derive(Clone)]
pub struct GpuBuffer {
    pub buffer: Buffer,
    pub length: usize,
}

impl GpuBuffer {
    fn new(data: &[f32]) -> Self {
        let ctx = &METAL_CTX;
        let size = (data.len() * std::mem::size_of::<f32>()) as u64;
        let buffer = ctx.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size,
            MTLResourceOptions::StorageModeShared,
        );
        Self { buffer, length: data.len() }
    }

    fn zeros(length: usize) -> Self {
        Self::new(&vec![0.0f32; length])
    }

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
    
    for (i, buf) in buffers.iter().enumerate() {
        encoder.set_buffer(i as u64, Some(*buf), 0);
    }
    let grid_size = MTLSize::new(length as u64, 1, 1);
    let tg_size = MTLSize::new(std::cmp::min(length as u64, 32).max(1), 1, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
}

// [修正]: 新增專門處理加總的 Dispatcher，負責傳遞 length 參數給 GPU
fn dispatch_sum(name: &str, in_buf: &Buffer, out_buf: &Buffer, length: usize) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function(name, None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    
    encoder.set_buffer(0, Some(in_buf), 0);
    encoder.set_buffer(1, Some(out_buf), 0);
    let len_u32 = length as u32;
    encoder.set_bytes(2, 4, &len_u32 as *const _ as *const _); // 這裡綁定 buffer(2)
    
    let grid_size = MTLSize::new(length as u64, 1, 1);
    let tg_size = MTLSize::new(std::cmp::min(length as u64, 32).max(1), 1, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
}

fn dispatch_matmul(name: &str, a: &Buffer, b: &Buffer, c: &Buffer, m: u32, k: u32, n: u32) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function(name, None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    
    encoder.set_buffer(0, Some(a), 0);
    encoder.set_buffer(1, Some(b), 0);
    encoder.set_buffer(2, Some(c), 0);
    encoder.set_bytes(3, 4, &m as *const _ as *const _);
    encoder.set_bytes(4, 4, &k as *const _ as *const _);
    encoder.set_bytes(5, 4, &n as *const _ as *const _);
    
    let grid_size = MTLSize::new(n as u64, m as u64, 1);
    let tg_size = MTLSize::new(8, 8, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
}

fn dispatch_transpose(in_buf: &Buffer, out_buf: &Buffer, rows: u32, cols: u32) {
    let ctx = &METAL_CTX;
    let func = ctx.library.get_function("transpose", None).unwrap();
    let pipeline = ctx.device.new_compute_pipeline_state_with_function(&func).unwrap();
    let cmd_buffer = ctx.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    
    encoder.set_buffer(0, Some(in_buf), 0);
    encoder.set_buffer(1, Some(out_buf), 0);
    encoder.set_bytes(2, 4, &rows as *const _ as *const _);
    encoder.set_bytes(3, 4, &cols as *const _ as *const _);
    
    let grid_size = MTLSize::new(cols as u64, rows as u64, 1);
    let tg_size = MTLSize::new(8, 8, 1);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
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
        Self(Rc::new(RefCell::new(TensorInner {
            data: GpuBuffer::new(data),
            grad: GpuBuffer::zeros(data.len()),
            shape: shape.to_vec(),
            op: Op::Leaf,
        })))
    }

    pub fn data(&self) -> Vec<f32> { self.0.borrow().data.to_vec() }
    pub fn grad(&self) -> Vec<f32> { self.0.borrow().grad.to_vec() }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let len = self.0.borrow().data.length;
        let out_buf = GpuBuffer::zeros(len);
        dispatch_1d("add", &[&self.0.borrow().data.buffer, &other.0.borrow().data.buffer, &out_buf.buffer], len);
        Self(Rc::new(RefCell::new(TensorInner {
            data: out_buf, grad: GpuBuffer::zeros(len), shape: self.0.borrow().shape.clone(), op: Op::Add(self.clone(), other.clone()),
        })))
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let len = self.0.borrow().data.length;
        let out_buf = GpuBuffer::zeros(len);
        dispatch_1d("mul", &[&self.0.borrow().data.buffer, &other.0.borrow().data.buffer, &out_buf.buffer], len);
        Self(Rc::new(RefCell::new(TensorInner {
            data: out_buf, grad: GpuBuffer::zeros(len), shape: self.0.borrow().shape.clone(), op: Op::Mul(self.clone(), other.clone()),
        })))
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let m = self.0.borrow().shape[0] as u32;
        let k = self.0.borrow().shape[1] as u32;
        let n = other.0.borrow().shape[1] as u32;
        let out_buf = GpuBuffer::zeros((m * n) as usize);
        dispatch_matmul("matmul", &self.0.borrow().data.buffer, &other.0.borrow().data.buffer, &out_buf.buffer, m, k, n);
        Self(Rc::new(RefCell::new(TensorInner {
            data: out_buf, grad: GpuBuffer::zeros((m * n) as usize), shape: vec![m as usize, n as usize], op: Op::Matmul(self.clone(), other.clone()),
        })))
    }

    pub fn relu(&self) -> Tensor {
        let len = self.0.borrow().data.length;
        let out_buf = GpuBuffer::zeros(len);
        dispatch_1d("relu", &[&self.0.borrow().data.buffer, &out_buf.buffer], len);
        Self(Rc::new(RefCell::new(TensorInner {
            data: out_buf, grad: GpuBuffer::zeros(len), shape: self.0.borrow().shape.clone(), op: Op::Relu(self.clone()),
        })))
    }

    pub fn sum(&self) -> Tensor {
        let len = self.0.borrow().data.length;
        let out_buf = GpuBuffer::zeros(1);
        // [修正]: 改用 dispatch_sum 傳遞 length
        dispatch_sum("sum_all", &self.0.borrow().data.buffer, &out_buf.buffer, len);
        Self(Rc::new(RefCell::new(TensorInner {
            data: out_buf, grad: GpuBuffer::zeros(1), shape: vec![1, 1], op: Op::Sum(self.clone()),
        })))
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let neg_ones = Tensor::new(&vec![-1.0; other.0.borrow().data.length], &other.0.borrow().shape);
        self.add(&other.mul(&neg_ones))
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(v: &Tensor, visited: &mut HashSet<Tensor>, topo: &mut Vec<Tensor>) {
            if !visited.contains(v) {
                visited.insert(v.clone());
                match &v.0.borrow().op {
                    Op::Add(a, b) | Op::Mul(a, b) | Op::Matmul(a, b) => {
                        build_topo(a, visited, topo);
                        build_topo(b, visited, topo);
                    }
                    Op::Relu(a) | Op::Sum(a) => { build_topo(a, visited, topo); }
                    Op::Leaf => {}
                }
                topo.push(v.clone());
            }
        }
        build_topo(self, &mut visited, &mut topo);

        // [修正]: 先取出 length，再進行借用與覆寫，避免同時 Immutable 與 Mutable Borrow！
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
                    let m = a.0.borrow().shape[0] as u32;
                    let k = a.0.borrow().shape[1] as u32;
                    let n = b.0.borrow().shape[1] as u32;

                    let b_t = GpuBuffer::zeros((n * k) as usize);
                    dispatch_transpose(&b.0.borrow().data.buffer, &b_t.buffer, k, n);
                    let a_grad_update = GpuBuffer::zeros((m * k) as usize);
                    dispatch_matmul("matmul", &grad.buffer, &b_t.buffer, &a_grad_update.buffer, m, n, k);
                    dispatch_1d("add_assign", &[&a.0.borrow().grad.buffer, &a_grad_update.buffer], (m * k) as usize);

                    let a_t = GpuBuffer::zeros((k * m) as usize);
                    dispatch_transpose(&a.0.borrow().data.buffer, &a_t.buffer, m, k);
                    let b_grad_update = GpuBuffer::zeros((k * n) as usize);
                    dispatch_matmul("matmul", &a_t.buffer, &grad.buffer, &b_grad_update.buffer, k, m, n);
                    dispatch_1d("add_assign", &[&b.0.borrow().grad.buffer, &b_grad_update.buffer], (k * n) as usize);
                }
                Op::Relu(a) => {
                    let tmp = GpuBuffer::zeros(grad.length);
                    dispatch_1d("relu_backward", &[&a.0.borrow().data.buffer, &grad.buffer, &tmp.buffer], grad.length);
                    dispatch_1d("add_assign", &[&a.0.borrow().grad.buffer, &tmp.buffer], grad.length);
                }
                Op::Sum(a) => {
                    // [修正]: 改用 dispatch_sum 傳遞 length
                    dispatch_sum("sum_backward", &a.0.borrow().grad.buffer, &grad.buffer, a.0.borrow().grad.length);
                }
                Op::Leaf => {}
            }
        }
    }
}