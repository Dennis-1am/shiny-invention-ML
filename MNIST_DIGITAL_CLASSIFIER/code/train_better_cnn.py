import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path
from models import BetterCNN

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
WEIGHTS_DIR = PROJECT_ROOT / 'weights'
WEIGHTS_DIR.mkdir(exist_ok=True)

# Setup Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Training on: {device}")

# Data Pipeline
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# Initialize Model
model = BetterCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

print("\n🎉 Training Complete!")

# Evaluation Block
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_set = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

model.eval()
correct = 0
total = 0

print("\n🧐 Evaluating on test data...")
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Save the trained weights
weights_path = WEIGHTS_DIR / 'mnist_better_cnn.pth'
torch.save(model.state_dict(), str(weights_path))
print(f"📂 Model saved as {weights_path}")

accuracy = 100 * correct / total
print(f"\n✅ Final Accuracy on 10,000 test images: {accuracy:.2f}%")
