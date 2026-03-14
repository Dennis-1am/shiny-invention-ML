import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
import torch.nn as nn

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

    def forward(self, x):
        return self.main(x)

# 2. Load the trained weights
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = BetterCNN().to(device)
model.load_state_dict(torch.load('code/weights/mnist_better_cnn.pth'))
model.eval()

# 3. Pre-process your "Real World" image
def predict_image(image_path):
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
    
    img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension [1, 1, 28, 28]

    with torch.no_grad():
        output = model(img_tensor)
        
    # Replace your current prediction print with this:
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    top_probs, top_indices = torch.topk(probabilities, 3)

    print("🤖 Model Analysis:")
    for i in range(3):
        print(f"   Rank {i+1}: Digit {top_indices[i].item()} ({top_probs[i].item()*100:.2f}%)")

# Run it!
predict_image('data/MNIST/test/#3.2.png')