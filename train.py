
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.optim as optim
from text_embedding_projection import text_features
from clip_similarity_projection import clip_features
from image_projection import image_features
from data_loader import get_labels
from text_and_clip_fusion import TransformerFusion
import copy


# ===============================
# **Early Stopping Class**
# ===============================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())  # Save best model
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience  # Stop if patience is exceeded



# ===============================
# **Dataset Class**
# ===============================        

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

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=5):
    print("Training...")
    model.train()
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        
        for batch in train_loader:
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

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total    

        # Validate Model after each epoch
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, return_loss=True)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
       

def evaluate(model, dataloader, device, return_loss=False):
    print("Evaluation...")
    model.eval()
    preds_list, labels_list = [], []
    total_loss, total = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            text = batch["text"].to(device)
            clip = batch["clip"].to(device)
            similarity = batch["similarity"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(text, clip, image, similarity)
            loss = criterion(outputs, labels)  # Compute loss

            preds = torch.argmax(outputs, dim=1)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = accuracy_score(labels_list, preds_list)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_list, preds_list, average="weighted")

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    if return_loss:
        return total_loss / len(dataloader), accuracy
    return accuracy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training data
train_file_path = "twitter/train.jsonl"
text_embeddings = text_features(train_file_path, device)  
clip_embeddings, similarity_scores = clip_features(train_file_path, device)  
image_embeddings = image_features(train_file_path, device)  
labels = get_labels(train_file_path).to(device)  


# Ensure consistent sizes across all modalities
min_size = min(text_embeddings.shape[0], clip_embeddings.shape[0], similarity_scores.shape[0], image_embeddings.shape[0], labels.shape[0])
text_embeddings = text_embeddings[:min_size]
clip_embeddings = clip_embeddings[:min_size]
similarity_scores = similarity_scores[:min_size]
image_embeddings = image_embeddings[:min_size]
labels = labels[:min_size]

# **Train-Validation Split (80-20)**
train_size = int(0.8 * min_size)
val_size = min_size - train_size

train_dataset, val_dataset = random_split(
    MultimodalDataset(text_embeddings, clip_embeddings, similarity_scores, image_embeddings, labels),
    [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load evaluation data
eval_file_path = "twitter/test.jsonl"
eval_text_embeddings = text_features(eval_file_path, device)
eval_clip_embeddings, eval_similarity_scores = clip_features(eval_file_path, device)
eval_image_embeddings = image_features(eval_file_path, device)
eval_labels = get_labels(eval_file_path)




# Ensure consistent sizes for evaluation
eval_min_size = min(eval_text_embeddings.shape[0], eval_clip_embeddings.shape[0], eval_similarity_scores.shape[0], eval_image_embeddings.shape[0], eval_labels.shape[0])
eval_text_embeddings = eval_text_embeddings[:eval_min_size]
eval_clip_embeddings = eval_clip_embeddings[:eval_min_size]
eval_similarity_scores = eval_similarity_scores[:eval_min_size] 
eval_image_embeddings = eval_image_embeddings[:eval_min_size]
eval_labels = eval_labels[:eval_min_size]

# Create evaluation dataset & dataloader
eval_dataset = MultimodalDataset(eval_text_embeddings, eval_clip_embeddings, eval_similarity_scores, eval_image_embeddings, eval_labels)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

# Initialize model, loss, and optimizer
model = TransformerFusion().to(device)  # Ensure the model handles three modalities
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Train the model
train(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=5)


# Evaluate the model on test data
evaluate(model, eval_loader, criterion, device)