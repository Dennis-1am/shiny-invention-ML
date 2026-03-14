import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from pathlib import Path
from models import BetterCNN

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
WEIGHTS_DIR = PROJECT_ROOT / 'weights'
WEIGHTS_DIR.mkdir(exist_ok=True)

def visualize_features(image_path, model_path):
    """Visualize the 32 feature maps from the first convolution layer"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = BetterCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess image
    img = Image.open(image_path).convert('L').resize((28, 28))
    img = ImageOps.invert(img)  # Only use if your source is black ink on white paper

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Extract the first layer (Conv2d)
    first_layer = model.main[0]

    # Pass the image through ONLY that first layer
    with torch.no_grad():
        feature_maps = first_layer(img_tensor)

    # Plot the 32 different "filters"
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    fig.suptitle(f"Internal Eyes: 32 Feature Maps for {image_path}")

    for i, ax in enumerate(axes.flat):
        # Extract the i-th filter's output
        f_map = feature_maps[0, i].cpu().numpy()
        ax.imshow(f_map, cmap='gray')
        ax.axis('off')

    plt.show()


if __name__ == '__main__':
    visualize_features(str(DATA_DIR / 'MNIST/test/#3.1.png'), str(WEIGHTS_DIR / 'mnist_better_cnn.pth'))
