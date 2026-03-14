import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torchvision.transforms as transforms

# 1. Re-define the architecture (must match exactly)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 13 * 13, 10)
        )
    def forward(self, x): return self.main(x)

class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x): return self.main(x)

def visualize_features(image_path, model_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = BetterCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess image (Same as your prediction script)
    img = Image.open(image_path).convert('L').resize((28, 28))
    img = ImageOps.invert(img) # Only use if your source is black ink on white paper
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 1. Extract the first layer (Conv2d)
    first_layer = model.main[0] 
    
    # 2. Pass the image through ONLY that first layer
    with torch.no_grad():
        feature_maps = first_layer(img_tensor)

    # 3. Plot the 32 different "filters"
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    fig.suptitle(f"Internal Eyes: 32 Feature Maps for {image_path}")
    
    for i, ax in enumerate(axes.flat):
        # Extract the i-th filter's output
        f_map = feature_maps[0, i].cpu().numpy() 
        ax.imshow(f_map, cmap='gray')
        ax.axis('off')

    plt.show()

# Run it
visualize_features('data/MNIST/test/#3.png', 'code/weights/mnist_better_cnn.pth')