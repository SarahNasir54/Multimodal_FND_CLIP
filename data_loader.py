import json
import re
from transformers import AutoTokenizer, AutoModel
import torch
import random
import os

def preprocess_post_text(post_text):
    post_text = re.sub(r'http\S+', '', post_text) #remore url
    post_text = re.sub(r'[#@]\S+', '', post_text) #remove # and @
    post_text = re.sub(r'[^a-zA-Z0-9\s]', '', post_text) #Remove special characters
    post_text = re.sub(r'\s+', ' ', post_text).strip() # remove spaces
    
    return post_text

def load_data(file_path, percentage=0.1):
    with open(file_path, "r", encoding="utf-8") as f:
        # Load the JSON data
        data = json.load(f)
        random.seed(42)
        sample_size = int(len(data) * percentage)
        sampled_data = random.sample(data, sample_size)  #only take 10% data
        for post in sampled_data:
            post["post_text"] = preprocess_post_text(post["post_text"])
        
        return sampled_data
    
def make_embeddings(text):
    pre_text = preprocess_post_text(text)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    model = AutoModel.from_pretrained("bert-base-multilingual-uncased")
    inputs = tokenizer(pre_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():  
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)

    return embedding
    
def find_image(image_name):
    for ext in image_extensions:
        image_path = f"{image_folder}/{image_name}{ext}"
        if os.path.exists(image_path):
            return image_path
    return None

def extract_features(path):
    processed_data = load_data(path)
    combined_features_list=[]
    for post in processed_data:
        id = post.get("post_id", "")
        text = post.get("post_text", "")
        friends =  post.get("num_friends", 0)
        followers = post.get("num_followers", 0)
        fratio = post.get("folfriend_ratio", 0)

        user_features = torch.tensor([
            followers,
            fratio,
            friends
            ], dtype=torch.float).unsqueeze(0)


        embedding = make_embeddings(text).unsqueeze(0)
        #print("Embedding shape:", embedding.unsqueeze(0).shape)

        combined_features = torch.cat([user_features, embedding], dim=1)
        print(combined_features.shape)
        #combined_features_list.append(combined_features)

    print("Combined features shape:", combined_features.shape)
    
    return combined_features

image_extensions = ['.jpg', '.jpeg', '.png', '.gif']    
image_folder = 'twitter/images'
file_path = "twitter/train.json"                    
#processed_data = extract_features(file_path)

# Display the processed data
# for post in processed_data:
#     #print(post)
#     print(json.dumps(post, indent=4, ensure_ascii=False))
