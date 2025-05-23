
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.optim as optim
from text_embedding_projection import text_features
from clip_similarity_projection import clip_features
from image_projection import image_features
from data_loader import get_labels
from text_and_clip_fusion import TransformerFusion
from sklearn.model_selection import train_test_split


class MultimodalDataset(Dataset):
    def __init__(self, text_features, clip_features, similarity_scores, image_features, labels):
        self.text_features = text_features
        self.clip_features = clip_features
        self.similarity_scores = similarity_scores 
        self.image_features = image_features
        self.labels = labels  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": self.text_features[idx],
            "clip": self.clip_features[idx],
            "similarity": self.similarity_scores[idx], 
            "image": self.image_features[idx],
            "label": self.labels[idx]
        }

def train(model, dataloader, criterion, optimizer, device, epochs=50):
    print("Training...")
    model.train()
    
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        
        for batch in dataloader:
            text = batch["text"].to(device)
            clip = batch["clip"].to(device)
            similarity = batch["similarity"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)

            # print(f"Text Shape: {text.shape}")       # Should be (batch_size, 1280)
            # print(f"CLIP Shape: {clip.shape}")       # Should be (batch_size, 1024)
            # print(f"Image Shape: {image.shape}")     # Should be (batch_size, 3072)
            # print(f"Similarity Shape: {similarity.shape}")  # Should be (batch_size, 1)

            optimizer.zero_grad()
            outputs = model(text, clip, image, similarity)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward(retain_graph=True)  # Backpropagation
            optimizer.step()  # Update parameters
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {correct/total:.4f}")
       

def evaluate(model, dataloader, device):
    print("Evaluation...")
    model.eval()
    preds_list, labels_list = [], []

    with torch.no_grad():
        for batch in dataloader:
            text = batch["text"].to(device)
            clip = batch["clip"].to(device)
            similarity = batch["similarity"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(text, clip, image, similarity)
            preds = torch.argmax(outputs, dim=1)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = accuracy_score(labels_list, preds_list)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_list, preds_list, average="weighted")

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "twitter/train.jsonl"
text_embeddings = text_features(file_path, device)
clip_embeddings, similarity_scores = clip_features(file_path, device)
image_embeddings = image_features(file_path, device)
labels = get_labels(file_path).to(device)

# Truncate to the smallest length to ensure alignment
min_size = min(text_embeddings.shape[0], clip_embeddings.shape[0], similarity_scores.shape[0], image_embeddings.shape[0], labels.shape[0])
text_embeddings = text_embeddings[:min_size]
clip_embeddings = clip_embeddings[:min_size]
similarity_scores = similarity_scores[:min_size]
image_embeddings = image_embeddings[:min_size]
labels = labels[:min_size]

# Perform train-test split
train_idx, test_idx = train_test_split(range(min_size), test_size=0.2, random_state=42, stratify=labels.cpu())

# Helper function to subset tensors
def subset(tensor, indices):
    return tensor[torch.tensor(indices)]

# Subset the data
train_dataset = MultimodalDataset(
    subset(text_embeddings, train_idx),
    subset(clip_embeddings, train_idx),
    subset(similarity_scores, train_idx),
    subset(image_embeddings, train_idx),
    subset(labels, train_idx)
)

test_dataset = MultimodalDataset(
    subset(text_embeddings, test_idx),
    subset(clip_embeddings, test_idx),
    subset(similarity_scores, test_idx),
    subset(image_embeddings, test_idx),
    subset(labels, test_idx)
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Initialize model, loss, and optimizer
model = TransformerFusion().to(device)  # Ensure the model handles three modalities
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
train(model, train_loader, criterion, optimizer, device, epochs=50)

# Evaluate the model on test data
evaluate(model, test_loader, device)
