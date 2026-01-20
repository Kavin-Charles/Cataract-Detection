
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

app = FastAPI(title="Cataract Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
# Model configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cataract_cnn_finetuned.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        print(f"Loading model from {MODEL_PATH}...")
        
        # Initialize ResNet18
        model = models.resnet18(pretrained=False)
        
        # Replace the fully connected layer for 2 classes (Cataract, Normal)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        # Load state dict
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        
        model = model.to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model globally on startup (or lazily)
model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['Normal', 'Cataract'] # Assuming 0: Normal, 1: Cataract - Need to verify class order if unknown, but usually alphabetical or 'Normal' first. 
# WAIT: If training data was ImageFolder, default is alphabetical: Cataract, Normal.
# Let's assume standard folder structure: 0: Cataract, 1: Normal.
# Re-checking logic: Usually "Cataract" comes before "Normal" alphabetically so 0=Cataract, 1=Normal.
# However, many datasets might be 0=Normal, 1=Cataract. 
# Safe bet for now: Return both probabilities or just label based on index.
# Let's check if we can verify this. For now I will return the raw prediction index and label map can be adjusted.
# Actually, I'll return both labels with probs.

CLASS_NAMES = ['Cataract', 'Normal'] # Alphabetical default for ImageFolder

@app.get("/")
async def root():
    return {"message": "Cataract Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        predicted_idx = predicted_class.item()
        confidence_score = confidence.item() * 100
        
        result = {
            "prediction": CLASS_NAMES[predicted_idx],
            "confidence": confidence_score,
            "has_cataract": CLASS_NAMES[predicted_idx] == 'Cataract',
            "probabilities": {
                CLASS_NAMES[0]: probabilities[0][0].item(),
                CLASS_NAMES[1]: probabilities[0][1].item()
            }
        }
        
        return JSONResponse(content=result)

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
