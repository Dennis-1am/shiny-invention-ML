import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
from pathlib import Path
from models import BetterCNN

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
WEIGHTS_DIR = PROJECT_ROOT / 'weights'
WEIGHTS_DIR.mkdir(exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = BetterCNN().to(device)
model.load_state_dict(torch.load(str(WEIGHTS_DIR / 'mnist_better_cnn.pth')))
model.eval()


def predict_image(image_path):
    """Predict digit from image and show top 3 predictions"""
    # Open image, convert to Grayscale ('L'), and resize to 28x28
    img = Image.open(image_path).convert('L').resize((28, 28))

    # MNIST images have white text on black background.
    # If you wrote with a black pen on white paper, we need to invert it!
    img = ImageOps.invert(img)
    img = img.filter(ImageFilter.MaxFilter(3))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(img_tensor)

    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    top_probs, top_indices = torch.topk(probabilities, 3)

    print("🤖 Model Analysis:")
    for i in range(3):
        print(f"   Rank {i+1}: Digit {top_indices[i].item()} ({top_probs[i].item()*100:.2f}%)")


if __name__ == '__main__':
    predict_image(str(DATA_DIR / 'MNIST/test/#3.1.png'))
