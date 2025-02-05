from PIL import Image
from data_loader import load_data
from data_loader import preprocess_post_text
from data_loader import find_image
from sentence_transformers import SentenceTransformer, util
import os
import torch

#image_folder = 'twitter/images'
train_file_path = "twitter/train.jsonl" 
test_file_path = "twitter/test.jsonl"
image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")


def prepare_clip_inputs(data):
    texts = []
    images = []
    item_count = 0

    for item in data:
        post_text = item.get('post_text')  
        post_text = preprocess_post_text(post_text)
        image_name = item.get('image')    
        image_path = find_image(image_name)
        if image_path:  
            try:
                image = Image.open(image_path).convert("RGB")
                texts.append(post_text)
                images.append(image)
                item_count += 1
            except Exception as e:
                print(f"Error loading image {image_name}: {e}")
    #print(f"Total items added: {item_count}")
    return texts, images

def compute_clip_similarity(texts, images):
    # Encode images and texts
    img_embeddings = model.encode(images, batch_size=8, convert_to_tensor=True)
    text_embeddings = model.encode(texts, batch_size=8, convert_to_tensor=True)

    clip_embeddings = torch.cat((img_embeddings, text_embeddings), dim=1)

    # Compute cosine similarity
    cos_scores = util.cos_sim(img_embeddings, text_embeddings)

    return text_embeddings, img_embeddings , clip_embeddings, cos_scores

def main(file_path):
    data = load_data(file_path, percentage=0.1)
    # Prepare text-image pairs
    texts, images = prepare_clip_inputs(data)

    if texts and images: 
        text_embeddings, image_embeddings, clip_embeddings, cos_scores = compute_clip_similarity(texts, images)
        print("text embedding matrix:")
        print(text_embeddings.shape)
        print("image embedding shape:", image_embeddings.shape)
        print("clip embedding shape:", clip_embeddings.shape)
        print("clip similarity score shape:", cos_scores.shape)
    else:
        print("No valid text-image pairs found.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main(train_file_path)

