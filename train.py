import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import AlexNet
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
EPOCHS = 16
BATCH_SIZE = 16
LEARNING_RATE = 0.0003

def train():
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training menggunakan device: {device}")
    
    # 3. Inisialisasi Model
    # Gunakan AlexNet yang sudah diadaptasi untuk gambar grayscale kecil
    model = AlexNet(input_channels=in_channels, num_classes=num_classes).to(device)
    print(model)
    
    # 4. Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # History
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    best_acc = 0.0
    
    print("\n--- Memulai Training ---")
    
    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # --- Perbaikan: pastikan label dalam format yang diharapkan loss ---
            if num_classes == 1:
                # BCEWithLogitsLoss expects shape (N, *) floats; buat menjadi (N,1)
                labels = labels.float().view(-1, 1)
            else:
                # CrossEntropyLoss expects target shape (N,) with class indices (long)
                # Jika label berbentuk one-hot (N, C) atau multi-dim, konversi ke indeks kelas
                if labels.ndim > 1:
                    # Pastikan tipe long lalu ambil argmax jika perlu
                    labels = labels.long()
                    if labels.shape[1] > 1:
                        labels = labels.argmax(dim=1)
                    else:
                        labels = labels.view(-1)
                else:
                    labels = labels.long()
            # --- end perbaikan ---

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Hitung akurasi
            if num_classes == 1:
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                # samakan dimensi untuk perbandingan
                train_total += labels.size(0)
                train_correct += (predicted.view(-1) == labels.view(-1)).sum().item()
            else:
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- Validasi ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Sama preprocessing label seperti di atas
                if num_classes == 1:
                    labels = labels.float().view(-1, 1)
                else:
                    if labels.ndim > 1:
                        labels = labels.long()
                        if labels.shape[1] > 1:
                            labels = labels.argmax(dim=1)
                        else:
                            labels = labels.view(-1)
                    else:
                        labels = labels.long()

                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                if num_classes == 1:
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted.view(-1) == labels.view(-1)).sum().item()
                else:
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        # Simpan model terbaik
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")

    print("\n--- Training Selesai ---")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    
    # Plot dan visualisasi
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == "__main__":
    train()
