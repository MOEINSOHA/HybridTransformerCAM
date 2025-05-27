import torch
from tqdm import tqdm

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    device = next(model.parameters()).device
    for epoch in range(num_epochs):
        model.train()
        total_loss_train = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()
        avg_train_loss = total_loss_train / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
        model.eval()
        total_loss_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss_val += loss.item()
        avg_val_loss = total_loss_val / len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
