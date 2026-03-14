import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm  # This is our progress bar!

# 1. Setup Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Training on: {device}")

# 2. Data Pipeline
# Augmentation for training (The "Gym")
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Clean for testing (The "Final Exam")
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# 3. Define the Model
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

model = BetterCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. The Training Loop
EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    # Wrap the loader in tqdm for a progress bar
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    
    for images, labels in loop:
        # Move data to Mac GPU (MPS)
        images, labels = images.to(device), labels.to(device)

        # The 4 Magic Steps of PyTorch
        optimizer.zero_grad()           # 1. Clear old math
        outputs = model(images)         # 2. Forward pass (guess)
        loss = criterion(outputs, labels) # 3. Calculate error
        loss.backward()                 # 4. Backward pass (math)
        optimizer.step()                # 5. Update weights

        # Update the progress bar text
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

print("\n🎉 Training Complete!")

# 5. The Evaluation Block
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

model.eval()  # Set model to evaluation mode
correct = 0
total = 0

print("\n🧐 Evaluating on test data...")
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # Get the index of the highest score
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Save the 'Furniture' (Weights) to a file
torch.save(model.state_dict(), 'code/weights/mnist_better_cnn.pth')
print("📂 Model saved as mnist_better_cnn.pth")

accuracy = 100 * correct / total
print(f"\n✅ Final Accuracy on 10,000 test images: {accuracy:.2f}%")
