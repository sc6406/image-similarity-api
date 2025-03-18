from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import clip
from PIL import Image
import io

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def calculate_similarity(img1, img2):
    image1 = preprocess(img1).unsqueeze(0).to(device)
    image2 = preprocess(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        features1 = model.encode_image(image1)
        features2 = model.encode_image(image2)

    features1 /= features1.norm(dim=-1, keepdim=True)
    features2 /= features2.norm(dim=-1, keepdim=True)

    similarity = (features1 @ features2.T).item()
    return similarity

@app.post("/compare-images/")
async def compare_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    img1 = Image.open(io.BytesIO(await image1.read())).convert('RGB')
    img2 = Image.open(io.BytesIO(await image2.read())).convert('RGB')

    similarity = calculate_similarity(img1, img2)

    if similarity > 0.9:
        message = "图片非常相似。"
    elif similarity > 0.7:
        message = "图片有一定相似度。"
    else:
        message = "图片不相似。"

    response = {
        "similarity_score": round(similarity, 4),
        "message": message
    }

    return JSONResponse(response)