"""
Modal.com Deployment for CNN-LSTM Action Recognition Model
API endpoint that serves the trained model for image annotation
"""

import modal
from pathlib import Path

# Initialize Modal app
app = modal.App(name="action-recognition-api")

# Get the directory where this script is located
LOCAL_DIR = Path(__file__).parent

# UCF101 Action Classes (101 classes)
ACTION_CLASSES = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
    "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
    "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
    "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
    "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "Hammering",
    "HammerThrow", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
    "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
    "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking", "Knitting",
    "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
    "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
    "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
    "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
    "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
    "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
    "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
    "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
    "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
    "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard",
    "YoYo"
]

# Define a Modal image with necessary dependencies and copy model file into it
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "fastapi",
        "python-multipart",
        "pydantic",
        "opencv-python-headless",
    )
    .add_local_file(
        local_path=str(LOCAL_DIR / "action_model (2).pth"),
        remote_path="/root/action_model.pth"
    )
)


# CNN-LSTM Model Definition
def get_model_class():
    import torch
    import torch.nn as nn
    from torchvision import models
    
    class CNN_LSTM(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            
            resnet = models.resnet50(weights="IMAGENET1K_V1")
            self.cnn = nn.Sequential(*list(resnet.children())[:-1])
            
            for p in self.cnn.parameters():
                p.requires_grad = False
            
            self.lstm = nn.LSTM(
                input_size=2048,
                hidden_size=512,
                num_layers=1,
                batch_first=True
            )
            
            self.fc = nn.Linear(512, num_classes)
        
        def forward(self, x):
            b, t, c, h, w = x.size()
            x = x.view(b * t, c, h, w)
            
            with torch.no_grad():
                features = self.cnn(x)
            
            features = features.squeeze(-1).squeeze(-1)
            features = features.view(b, t, -1)
            
            lstm_out, _ = self.lstm(features)
            return self.fc(lstm_out[:, -1])
    
    return CNN_LSTM


# Create a standalone predict function that uses the model
@app.function(image=image, gpu="t4")
def predict_image(image_data: bytes) -> dict:
    """Predict action class from image bytes"""
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    import io
    import cv2
    import numpy as np
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Get the model class
        CNN_LSTM = get_model_class()
        
        # Initialize model architecture
        num_classes = len(ACTION_CLASSES)
        model = CNN_LSTM(num_classes)
        
        # Load weights from the model file embedded in the image
        model_path = "/root/action_model.pth"
        print(f"Loading model from: {model_path}")
        
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully. Number of classes: {num_classes}")
        
        model = model.to(device)
        model.eval()
        
        # Setup transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load and preprocess image
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_np = np.array(img)
        img_resized = cv2.resize(img_np, (224, 224))
        
        # Apply transform and create tensor with shape (batch, seq_len, channels, height, width)
        # For single image, we use seq_len=1
        image_tensor = transform(img_resized).unsqueeze(0).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = outputs.argmax(1).item()
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        top5_predictions = {
            ACTION_CLASSES[idx.item()]: float(prob.cpu())
            for prob, idx in zip(top5_probs, top5_indices)
        }
        
        return {
            "prediction": ACTION_CLASSES[predicted_class],
            "confidence": float(probabilities[predicted_class].cpu()),
            "top5_predictions": top5_predictions,
            "all_classes_count": num_classes
        }
    except Exception as e:
        import traceback
        return {"error": f"Error processing image: {str(e)}", "traceback": traceback.format_exc()}


# Annotate image with prediction
@app.function(image=image, gpu="t4")
def annotate_image(image_data: bytes) -> dict:
    """Predict action and return annotated image"""
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    import io
    import cv2
    import numpy as np
    import base64
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get the model class
        CNN_LSTM = get_model_class()
        
        # Initialize model
        num_classes = len(ACTION_CLASSES)
        model = CNN_LSTM(num_classes)
        
        model_path = "/root/action_model.pth"
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Setup transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load original image
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_np = np.array(img)
        original = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Preprocess for model
        img_resized = cv2.resize(img_np, (224, 224))
        image_tensor = transform(img_resized).unsqueeze(0).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = outputs.argmax(1).item()
        
        label = ACTION_CLASSES[predicted_class]
        confidence = float(probabilities[predicted_class].cpu())
        
        # Annotate the image
        h, w = original.shape[:2]
        box_height = max(60, int(h * 0.08))
        
        cv2.rectangle(original, (10, 10), (min(w - 10, 500), box_height + 10), (0, 0, 0), -1)
        
        font_scale = max(0.6, min(1.2, w / 500))
        cv2.putText(
            original, 
            f"Action: {label} ({confidence*100:.1f}%)", 
            (20, box_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (0, 255, 0), 
            2
        )
        
        # Encode annotated image to base64
        _, buffer = cv2.imencode('.jpg', original)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        top5_predictions = {
            ACTION_CLASSES[idx.item()]: float(prob.cpu())
            for prob, idx in zip(top5_probs, top5_indices)
        }
        
        return {
            "prediction": label,
            "confidence": confidence,
            "top5_predictions": top5_predictions,
            "annotated_image": annotated_base64
        }
    except Exception as e:
        import traceback
        return {"error": f"Error processing image: {str(e)}", "traceback": traceback.format_exc()}


# Mount the FastAPI app to Modal
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    
    web_app = FastAPI(title="Action Recognition API")
    
    # Add CORS middleware to allow requests from any origin
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @web_app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        """
        Upload an image and get action classification
        """
        try:
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
                return {"error": "File must be an image (JPG, PNG, or WebP)"}
            
            image_data = await file.read()
            result = predict_image.remote(image_data)
            return result
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    @web_app.post("/annotate")
    async def annotate(file: UploadFile = File(...)):
        """
        Upload an image and get annotated image with action classification
        """
        try:
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
                return {"error": "File must be an image (JPG, PNG, or WebP)"}
            
            image_data = await file.read()
            result = annotate_image.remote(image_data)
            return result
        except Exception as e:
            return {"error": f"Annotation failed: {str(e)}"}
    
    @web_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "model": "CNN-LSTM Action Recognition (UCF101)"}
    
    @web_app.get("/classes")
    async def get_classes():
        """Get available action classes"""
        return {"classes": ACTION_CLASSES, "count": len(ACTION_CLASSES)}
    
    @web_app.get("/", response_class=HTMLResponse)
    async def root():
        """API documentation page"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Action Recognition API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #333; }
                .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                code { background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>🎬 Action Recognition API</h1>
            <p>CNN-LSTM model trained on UCF101 dataset for action recognition in images.</p>
            
            <h2>Endpoints:</h2>
            
            <div class="endpoint">
                <h3>POST /predict</h3>
                <p>Upload an image to get action classification.</p>
                <p>Request: <code>multipart/form-data</code> with <code>file</code> field</p>
            </div>
            
            <div class="endpoint">
                <h3>POST /annotate</h3>
                <p>Upload an image to get annotated image with action label.</p>
                <p>Returns prediction + base64 encoded annotated image.</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /classes</h3>
                <p>Get list of all 101 action classes.</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /health</h3>
                <p>Health check endpoint.</p>
            </div>
        </body>
        </html>
        """
    
    return web_app


if __name__ == "__main__":
    print("=" * 60)
    print("Action Recognition Model - Modal.com Deployment")
    print("=" * 60)
    print("\nTo deploy this app to Modal, run:")
    print("    modal deploy action_recognition_modal.py")
    print("\nTo run locally for testing:")
    print("    modal serve action_recognition_modal.py")
    print("=" * 60)
