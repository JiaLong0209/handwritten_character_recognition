# === main.py (FastAPI server) ===
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from efficientnet_pytorch import EfficientNet
from PIL import Image
import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import uvicorn
import json

# Load class mapping from JSON
with open("class_mapping.json", "r", encoding="utf-8") as f:
    class_mapping = json.load(f)

# Convert index -> class
index_to_class = {v: k for k, v in class_mapping.items()}

# Load model and hyperparameters
model_dir = "training_results"
pth_name = "best_model_checkpoint.pth"
model_path = os.path.join(model_dir, pth_name)
hyperparams_path = os.path.join(model_dir, "best_hyperparameters.json")

# Load hyperparameters
try:
    with open(hyperparams_path, "r", encoding="utf-8") as f:
        hyperparams = json.load(f)
    print("Loaded hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
except FileNotFoundError:
    print(f"Warning: Hyperparameters file not found at {hyperparams_path}")
    hyperparams = {
        'dropout_rate': 0.4329770563201687  # Fallback to default value
    }
except json.JSONDecodeError:
    print(f"Warning: Could not parse hyperparameters file at {hyperparams_path}")
    hyperparams = {
        'dropout_rate': 0.4329770563201687  # Fallback to default value
    }

print(f"Loading model from: {model_path}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dict_model(model_path=''):
    """
    Load model from checkpoint file
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get number of classes from class mapping
    class_num = len(class_mapping)
    
    # Create model with same architecture
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(hyperparams['dropout_rate']),  # Use dropout rate from hyperparameters
        nn.Linear(num_ftrs, class_num)
    )
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully from: {model_path}")
    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Using dropout rate: {hyperparams['dropout_rate']}")
    print(f"Checkpoint info:")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Best accuracy: {checkpoint['acc']:.4f}")
    print(f"  - Loss: {checkpoint['loss']:.4f}")
    
    return model

def load_model(model_path=''):
    """
    Load the saved model directly
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the entire model
    model = torch.load(model_path, map_location=device, weights_only = False)
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully from: {model_path}")
    print(f"Model is on device: {next(model.parameters()).device}")
    
    return model

model = load_dict_model(model_path)
# model = load_model(model_path)
# model = load_model_safe_globals(model_path)

print(f'model: {model}')

# Image transform
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return HTMLResponse(open("main.html", "r", encoding="utf-8").read())

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probs, 0)

    result = {
        "class": index_to_class[predicted_idx.item()],
        "confidence": round(confidence.item(), 4)
    }
    return JSONResponse(content=result)

# Entry point
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
