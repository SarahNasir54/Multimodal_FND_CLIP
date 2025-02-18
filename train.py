
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.optim as optim
from text_embedding_projection import text_features
from clip_similarity_projection import clip_features
from data_loader import get_labels
from text_and_clip_fusion import TransformerFusion


class MultimodalDataset(Dataset):
    def __init__(self, text_features, clip_features, labels):
        self.text_features = text_features
        self.clip_features = clip_features
        self.labels = labels  # Tensor of shape [batch_size]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": self.text_features[idx],
            "clip": self.clip_features[idx],
            "label": self.labels[idx]
        }

def train(model, dataloader, criterion, optimizer, device, epochs=5):
    model.train()
    
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        
        for batch in dataloader:
            text, clip, labels = batch["text"].to(device), batch["clip"].to(device), batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(text, clip)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {correct/total:.4f}")

def evaluate(model, dataloader, device):
    model.eval()
    preds_list, labels_list = [], []

    with torch.no_grad():
        for batch in dataloader:
            text, clip, labels = batch["text"].to(device), batch["clip"].to(device), batch["label"].to(device)
            outputs = model(text, clip)
            preds = torch.argmax(outputs, dim=1)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = accuracy_score(labels_list, preds_list)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_list, preds_list, average="weighted")

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
file_path = "twitter/train.jsonl"
text_embeddings = text_features(file_path)
clip_embeddings = clip_features(file_path)
labels = get_labels(file_path)

min_size = min(text_embeddings.shape[0], clip_embeddings.shape[0], labels.shape[0])
text_embeddings = text_embeddings[:min_size]
clip_embeddings = clip_embeddings[:min_size]
labels = labels[:min_size]

# Create dataset & dataloader
dataset = MultimodalDataset(text_embeddings, clip_embeddings, labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = TransformerFusion().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Train the model
train(model, train_loader, criterion, optimizer, device, epochs=5)

# Evaluate the model
evaluate(model, train_loader, device)