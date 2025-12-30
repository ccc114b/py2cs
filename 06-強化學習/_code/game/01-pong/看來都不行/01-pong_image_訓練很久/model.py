import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    """
    自動選擇最佳的計算設備
    優先順序: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

class DQN(nn.Module):
    """Deep Q-Network for Pong"""
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        # 卷積層處理圖像輸入
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 計算卷積層輸出的尺寸
        conv_out_size = self._get_conv_output(input_shape)
        
        # 全連接層
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def _get_conv_output(self, shape):
        """計算卷積層輸出的大小"""
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            dummy = self.conv3(dummy)
            return int(dummy.numel())
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    """經驗回放緩衝區"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)