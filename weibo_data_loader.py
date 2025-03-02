import json
import re
import torch
import random
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from googletrans import Translator
import asyncio

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
    image_folder = 'weibo/images'
    for ext in image_extensions:
        image_path = f"{image_folder}/{image_name}{ext}"
        if os.path.exists(image_path):
            return image_path
    return None

def load_data(file_path, percentage=0.1):

    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    random.seed(42)
    sample_size = int(len(data) * percentage)
    sampled_data = random.sample(data, sample_size)

    print(f"Loaded {len(sampled_data)} samples")
    return data

async def translate_text(text, translator):
    """Asynchronously translate text from Chinese to English."""
    try:
        translation = await translator.translate(text, src="zh-cn", dest="en")
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

async def translate_data(file_path):
    data = load_data(file_path)

    translator = Translator()

    tasks = [translate_text(item["text"], translator) for item in data]
    translations = await asyncio.gather(*tasks)

    # Assign translated texts to data
    for item, translated_text in zip(data, translations):
        item["translated_text"] = translated_text

    print(f"Loaded and translated {len(data)} samples")

    with open("weibo/test_translated.jsonl", "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return data

def run_translation(file_path):
    return asyncio.run(translate_data(file_path))

def load_weibo_data(file_path, percentage=0.1):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    random.seed(42)
    sample_size = int(len(data) * percentage)
    sampled_data = random.sample(data, sample_size)

    # Preprocess text
    for post in sampled_data:
        post["translated_text"] = preprocess_post_text(post["translated_text"])

    print(f"Loaded {len(sampled_data)} samples")

    return sampled_data

def make_embeddings(file_path):
    """Generate BERT embeddings for text using CUDA."""
    #print("Making BERT embeddings...")
    processed_data = load_data(file_path)

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
    model = AutoModel.from_pretrained("google-bert/bert-base-chinese").to(device)
    model.eval() 

    all_embeddings = []
    
    for post in processed_data:
        text = post.get("text", "")
        pre_text = preprocess_post_text(text)

        inputs = tokenizer(pre_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to(device)

        with torch.no_grad():  
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)  
            all_embeddings.append(embedding)

    final_embeddings = torch.stack(all_embeddings)  
    print(f"Final BERT Embeddings Shape: {final_embeddings.shape}") 

    return final_embeddings

def main(file_path):
    embeddings = make_embeddings(file_path)
    #print("data", data)

if __name__ == "__main__":
    file_path = "weibo/train_translated.jsonl"
    main(file_path)