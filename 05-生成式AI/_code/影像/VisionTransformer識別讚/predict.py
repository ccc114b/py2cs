import torch
from torchvision import datasets, transforms
from model import ViT, get_device
import matplotlib.pyplot as plt

DEVICE = get_device()

def predict():
    # 預處理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 載入測試集
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 載入模型
    model = ViT().to(DEVICE)
    model.load_state_dict(torch.load("vit_mnist.pth", map_location=DEVICE))
    model.eval()

    # 取一張圖預測
    data, target = next(iter(test_loader))
    with torch.no_grad():
        output = model(data.to(DEVICE))
        prediction = output.argmax(dim=1, keepdim=True).item()

    print(f"真實數字: {target.item()}, 模型預測: {prediction}")

    # 顯示圖片
    plt.imshow(data.squeeze(), cmap='gray')
    plt.title(f"Prediction: {prediction}")
    plt.show()

if __name__ == "__main__":
    predict()