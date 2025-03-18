import torch
import clip
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess images
image1 = preprocess(Image.open("product_reference.png")).unsqueeze(0).to(device)
image2 = preprocess(Image.open("alibaba_product.png")).unsqueeze(0).to(device)

# Encode images
with torch.no_grad():
    image_features1 = model.encode_image(image1)
    image_features2 = model.encode_image(image2)

# Normalize the feature vectors
image_features1 /= image_features1.norm(dim=-1, keepdim=True)
image_features2 /= image_features2.norm(dim=-1, keepdim=True)

# Compute cosine similarity
similarity = (image_features1 @ image_features2.T).item()

# Print the similarity score
print(f"\n✅ 相似度得分: {similarity:.4f}")

# Interpret the result
if similarity > 0.9:
    print("🟢 图片很他妈的相似!")
elif similarity > 0.7:
    print("🟡 图片算是相似.")
else:
    print("🔴 图片一点都不像.")