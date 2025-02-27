from PIL import Image
from data_loader import load_data, preprocess_post_text, find_image
from sentence_transformers import SentenceTransformer, util
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load CLIP Model on GPU
model = SentenceTransformer("sentence-transformers/clip-ViT-B-32").to(device)

train_file_path = "twitter/train.jsonl" 
test_file_path = "twitter/test.jsonl"

# ===============================
# **Prepare CLIP Input Pairs**
# ===============================
def prepare_clip_inputs(data):
    #print("Preparing inputs for CLIP...")
    texts = []
    images = []
    successful_images = 0

    for item in data:
        post_text = item.get('post_text')
        post_text = preprocess_post_text(post_text) 
        tokens = post_text.split()[:77]  # Truncate to 77 tokens
        short_text = " ".join(tokens) 
        image_name = item.get('image')    
        image_path = find_image(image_name) 

        if image_path:  
            image = Image.open(image_path).convert("RGB")
            texts.append(short_text)
            images.append(image)
            successful_images += 1

    #print(f"Successfully Loaded Image-Text Pairs: {successful_images}")
    #print(f"Total Texts: {len(texts)}, Total Images: {len(images)}")
    return texts, images  


def compute_clip_similarity(texts, images):
    #print("Computing CLIP similarity...")

    img_embeddings = model.encode(images, batch_size=32, convert_to_tensor=True).to(device)
    text_embeddings = model.encode(texts, batch_size=32, convert_to_tensor=True).to(device)

    clip_embeddings = torch.cat((img_embeddings, text_embeddings), dim=1)

    cos_scores = util.cos_sim(img_embeddings, text_embeddings).to(device)

    return text_embeddings, img_embeddings, clip_embeddings, cos_scores

# def main(file_path,device):
#     print("Computing CLIP similarity...")
#     data = load_data(file_path, percentage=0.1) 

#     texts, images = prepare_clip_inputs(data)

#     text_embeddings, image_embeddings, clip_embeddings, cos_scores = compute_clip_similarity(texts, images)

#     print("Text Embedding Shape:", text_embeddings.shape)
#     print("Image Embedding Shape:", image_embeddings.shape)
#     print("CLIP Embedding Shape:", clip_embeddings.shape)
#     print("CLIP Similarity Score Shape:", cos_scores.shape)


# if __name__ == "__main__":
#     import warnings
#     warnings.filterwarnings("ignore")
#     main(test_file_path,device)
