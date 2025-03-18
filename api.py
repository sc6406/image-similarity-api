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

@app.post("/check_similarity/")
async def compare_image(file: UploadFile = File(...)):
    preset_image = Image.open("product_reference.png").convert('RGB')
    uploaded_image = Image.open(io.BytesIO(await file.read())).convert('RGB')

    similarity = calculate_similarity(preset_image, uploaded_image)

    if similarity > 0.9:
        message = "图片非常相似。"
    else:
        message = "图片不够相似。"

    return JSONResponse({"similarity": similarity, "message": message})