import json
import re
from transformers import AutoTokenizer, AutoModel
import torch
import random
import os
from sklearn.preprocessing import LabelEncoder



def preprocess_post_text(post_text):
    post_text = re.sub(r'http\S+', '', post_text) #remore url
    post_text = re.sub(r'[#@]\S+', '', post_text) #remove # and @
    post_text = re.sub(r'[^a-zA-Z0-9\s]', '', post_text) #Remove special characters
    post_text = re.sub(r'\s+', ' ', post_text).strip() # remove spaces
    
    return post_text

def load_data(file_path, percentage=0.1):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f] 
                        
    random.seed(42)
    sample_size = int(len(data) * percentage)
    sampled_data = random.sample(data, sample_size)

    # Preprocess text
    for post in sampled_data:
        post["post_text"] = preprocess_post_text(post["post_text"])
        
    return sampled_data
    
def find_image(image_name):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif'] 
    image_folder = 'twitter/images'
    for ext in image_extensions:
        image_path = f"{image_folder}/{image_name}{ext}"
        if os.path.exists(image_path):
            return image_path
    return None
    
def make_embeddings(file_path):
    processed_data = load_data(file_path)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    model = AutoModel.from_pretrained("bert-base-multilingual-uncased")

    for post in processed_data:
        text = post.get("post_text", "")  # Extract text
        pre_text = preprocess_post_text(text)
        inputs = tokenizer(pre_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        with torch.no_grad():  
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)

            embedding = embedding.unsqueeze(0)
            #print(embedding.shape)

    return embedding
    
def get_labels(file_path):
    data = load_data(file_path)
    all_labels = [post.get("label", "") for post in data]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels) 

    encoded_labels = torch.tensor(encoded_labels, dtype=torch.long) 

    #print(label_encoder.classes_)  
    #print(encoded_labels)

    return encoded_labels
                
def main(file_path):
    
    #embedding = make_embeddings(file_path)
    #print("Embedding features shape:", embedding.shape)

    labels = get_labels(file_path)

    
if __name__ == "__main__":
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']    
    image_folder = 'twitter/images'
    file_path = "twitter/train.jsonl"
    #main(file_path)
