import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# ==== 1️⃣ MODEL TANPA CNN ====
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden1=256, hidden2=128, num_classes=2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten gambar
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==== 2️⃣ KONFIGURASI ====
DEVICE = 'cpu'
NUM_CLASSES = 2
BATCH_SIZE = 10

# Transformasi sederhana (grayscale + resize)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Dataset dummy (nggak download apa pun)
test_dataset = datasets.FakeData(
    size=10,
    image_size=(1, 28, 28),
    num_classes=NUM_CLASSES,
    transform=transform
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== 3️⃣ INISIALISASI MODEL ====
model = SimpleMLP(num_classes=NUM_CLASSES).to(DEVICE)
model.eval()

# ==== 4️⃣ PREDIKSI ====
images, labels = next(iter(test_loader))
outputs = model(images)
probs = F.softmax(outputs, dim=1)
_, preds = torch.max(probs, 1)

# ==== 5️⃣ VISUALISASI ====
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i, ax in enumerate(axes):
    img = images[i].squeeze().detach().numpy()
    pred_label = preds[i].item()
    true_label = labels[i].item()
    prob = probs[i][pred_label].item()

    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.text(0, -2, f"Pred: {pred_label}", color='royalblue', fontsize=9, weight='bold')
    ax.text(0, 2, f"Prob: {prob:.2f}", color='darkorange', fontsize=9, weight='bold')
    ax.text(0, 26, f"GT: {true_label}", color='green', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig("mlp_predictions.png")
plt.show()

print("✅ Hasil disimpan sebagai: mlp_predictions.png")
