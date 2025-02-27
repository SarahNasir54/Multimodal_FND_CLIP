import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import make_embeddings, load_data
from clip_similarity import compute_clip_similarity, prepare_clip_inputs
from transformers import CLIPTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_dim = 768
clip_dim = 512
hidden_dim = 256


def load_clip_embedding(file_path):
    data = load_data(file_path, percentage=0.3)
    # Prepare text-image pairs
    texts, images = prepare_clip_inputs(data)

    if texts and images: 
        text_embeddings, _, _, _ = compute_clip_similarity(texts, images)

    return text_embeddings



# ===============================
# **Process Text & CLIP Features**
# ===============================

def text_features(file_path, device):
    #print("Computing Text Features...")
    bert_embedding = make_embeddings(file_path).to(device)
    clip_text_embedding = load_clip_embedding(file_path)

    # Ensure embeddings have same batch size
    min_size = min(bert_embedding.shape[0], clip_text_embedding.shape[0])
    bert_embedding = bert_embedding[:min_size]
    clip_text_embedding = clip_text_embedding[:min_size]

    concatenated_embedding = torch.cat([bert_embedding, clip_text_embedding], dim=1)

    return concatenated_embedding 

# ===============================
# **Main Function**
# ===============================

# def main():
#     print("Starting processing...")
#     file_path = "twitter/train.jsonl"
    
#     text_embeddings = text_features(file_path, device)

#     if text_embeddings is None:
#         print("text_features() returned None!")
#     else:
#         print(f"text_features() returned shape: {text_embeddings.shape}")

# # ===============================
# # **Run Script**
# # ===============================

# if __name__ == "__main__":
#     main()
