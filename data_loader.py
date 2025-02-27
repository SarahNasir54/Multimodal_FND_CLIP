import json
import re
import torch
import random
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_post_text(post_text):
    """Clean text: remove URLs, hashtags, mentions, special chars, and extra spaces."""
    post_text = re.sub(r'http\S+', '', post_text)  # Remove URLs
    post_text = re.sub(r'[#@]\S+', '', post_text)  # Remove hashtags and mentions
    post_text = re.sub(r'[^a-zA-Z0-9\s]', '', post_text)  # Remove special characters
    post_text = re.sub(r'\s+', ' ', post_text).strip()  # Remove extra spaces
    return post_text

def find_image(image_name):
    """Finds image file with supported extensions."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_folder = 'twitter/images'
    for ext in image_extensions:
        image_path = f"{image_folder}/{image_name}{ext}"
        if os.path.exists(image_path):
            return image_path
    return None

def load_data(file_path, percentage=0.3):
    #print("Loading data...")
    """Load JSONL data and filter to ensure text and image match."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    random.seed(42)
    sample_size = int(len(data) * percentage)
    sampled_data = random.sample(data, sample_size)

    # Preprocess text
    for post in sampled_data:
        post["post_text"] = preprocess_post_text(post["post_text"])

    #print(f"Loaded {len(sampled_data)} samples")
    return sampled_data
    
def make_embeddings(file_path):
    """Generate BERT embeddings for text using CUDA."""
    #print("Making BERT embeddings...")
    processed_data = load_data(file_path)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    model = AutoModel.from_pretrained("bert-base-multilingual-uncased").to(device)
    model.eval() 

    all_embeddings = []
    
    for post in processed_data:
        text = post.get("post_text", "")
        pre_text = preprocess_post_text(text)

        inputs = tokenizer(pre_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to(device)

        with torch.no_grad():  
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)  
            all_embeddings.append(embedding)

    final_embeddings = torch.stack(all_embeddings)  
    #print(f"Final BERT Embeddings Shape: {final_embeddings.shape}") 

    return final_embeddings

def get_labels(file_path):
    """Encode labels into tensor format for classification."""
    data = load_data(file_path)  
    all_labels = [post.get("label", "") for post in data]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)  
    encoded_labels = torch.tensor(encoded_labels, dtype=torch.long, device=device)  # Move labels to CUDA

    return encoded_labels
                
# def main(file_path):
#     """Main function to generate embeddings and labels on CUDA."""
#     embeddings = make_embeddings(file_path)
#     print("Embedding features shape:", embeddings.shape)  # Expect `[1271, 768]` (same as CLIP)

#     labels = get_labels(file_path)
#     print("Labels shape:", labels.shape)

# if __name__ == "__main__":
#     #file_path = "twitter/train.jsonl"
#     file_path = "twitter/test.jsonl"
#     main(file_path)
