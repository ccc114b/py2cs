import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)
        self.grad = np.zeros(self.data.shape)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self): return self.data.shape

    # 在 Tensor 類別中新增一個輔助方法
    def _reduce_grad(self, grad, target_shape):
        # 1. 處理多出來的維度 (例如把 3D 壓回 2D)
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        # 2. 處理原本是 1 但被廣播變大的維度 (例如 (1, 32) 變 (8, 32))
        for axis, size in enumerate(target_shape):
            if size == 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # 使用輔助函式縮減梯度
            self.grad += self._reduce_grad(out.grad, self.shape)
            other.grad += self._reduce_grad(out.grad, other.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            # 計算原始梯度
            g_self = other.data * out.grad
            g_other = self.data * out.grad
            # 縮減梯度後累加
            self.grad += self._reduce_grad(g_self, self.shape)
            other.grad += self._reduce_grad(g_other, other.shape)
        out._backward = _backward
        return out

    """
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.broadcast_to(other, self.shape))
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            # 處理 broadcasting 的梯度累加
            self_grad = out.grad
            while self_grad.ndim > self.data.ndim: self_grad = self_grad.sum(axis=0)
            for axis, size in enumerate(self.data.shape):
                if size == 1: self_grad = self_grad.sum(axis=axis, keepdims=True)
            self.grad += self_grad

            other_grad = out.grad
            while other_grad.ndim > other.data.ndim: other_grad = other_grad.sum(axis=0)
            for axis, size in enumerate(other.data.shape):
                if size == 1: other_grad = other_grad.sum(axis=axis, keepdims=True)
            other.grad += other_grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.broadcast_to(other, self.shape))
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    """

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    """
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), (self, other), 'matmul')
        def _backward():
            self.grad += np.matmul(out.grad, other.data.swapaxes(-1, -2))
            other.grad += np.matmul(self.data.swapaxes(-1, -2), out.grad)
        out._backward = _backward
        return out
    """

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), (self, other), 'matmul')

        def _backward():
            # 計算原始梯度
            grad_self = np.matmul(out.grad, other.data.swapaxes(-1, -2))
            grad_other = np.matmul(self.data.swapaxes(-1, -2), out.grad)

            # --- 修正廣播造成的維度不一致 ---
            
            # 1. 處理 self 的梯度
            # 如果 grad_self 比 self.data 多了維度（例如 batch 維度），就把它們加總
            while grad_self.ndim > self.data.ndim:
                grad_self = grad_self.sum(axis=0)
            # 處理像 (1, 10) 廣播成 (5, 10) 的情況
            for axis, size in enumerate(self.data.shape):
                if size == 1:
                    grad_self = grad_self.sum(axis=axis, keepdims=True)
            self.grad += grad_self

            # 2. 處理 other 的梯度
            while grad_other.ndim > other.data.ndim:
                grad_other = grad_other.sum(axis=0)
            for axis, size in enumerate(other.data.shape):
                if size == 1:
                    grad_other = grad_other.sum(axis=axis, keepdims=True)
            other.grad += grad_other
            
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')
        def _backward():
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    def transpose(self, axis1, axis2):
        out = Tensor(self.data.swapaxes(axis1, axis2), (self,), 'transpose')
        def _backward():
            self.grad += out.grad.swapaxes(axis1, axis2)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            grad = out.grad
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        # 數值穩定版 softmax
        exps = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        probs = exps / np.sum(exps, axis=axis, keepdims=True)
        out = Tensor(probs, (self,), 'softmax')
        def _backward():
            # 簡化的 softmax backward 適用於跟 cross_entropy 結合，
            # 這裡實作通用版梯度
            for i in range(out.grad.shape[0]): # 簡化處理
                s = probs[i].reshape(-1, 1)
                jac = np.diagflat(probs[i]) - np.dot(s, s.T)
                self.grad[i] += np.dot(jac, out.grad[i])
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1