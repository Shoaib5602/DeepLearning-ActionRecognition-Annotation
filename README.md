# Action Recognition Model - Modal.com Deployment Guide

## 📋 Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install and authenticate

```bash
pip install modal
modal token new
```

## 🚀 Deployment Steps

### Step 1: Deploy to Modal.com

Open terminal in this folder and run:

```bash
modal deploy action_recognition_modal.py
```

**Expected Output:**
```
✓ Created objects.
├── 🔨 Created predict_image.
├── 🔨 Created annotate_image.
└── 🔨 Created fastapi_app => https://YOUR_USERNAME--action-recognition-api-fastapi-app.modal.run
```

### Step 2: Copy Your API URL

After deployment, Modal will show your API URL. It looks like:
```
https://YOUR_USERNAME--action-recognition-api-fastapi-app.modal.run
```

### Step 3: Use the UI

1. Open `action_recognition_ui.html` in your browser
2. Paste your Modal API URL in the "API Endpoint URL" field
3. Upload an image and click "Recognize Action" or "Annotate Image"

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation |
| `/health` | GET | Health check |
| `/classes` | GET | List all 101 action classes |
| `/predict` | POST | Predict action from image |
| `/annotate` | POST | Predict + return annotated image |

### Example API Usage (Python)

```python
import requests

API_URL = "https://YOUR_USERNAME--action-recognition-api-fastapi-app.modal.run"

# Predict action
with open("image.jpg", "rb") as f:
    response = requests.post(f"{API_URL}/predict", files={"file": f})
    print(response.json())

# Annotate image
with open("image.jpg", "rb") as f:
    response = requests.post(f"{API_URL}/annotate", files={"file": f})
    result = response.json()
    
    # Save annotated image
    import base64
    with open("annotated.jpg", "wb") as out:
        out.write(base64.b64decode(result["annotated_image"]))
```

### Example API Usage (cURL)

```bash
# Health check
curl https://YOUR_USERNAME--action-recognition-api-fastapi-app.modal.run/health

# Get classes
curl https://YOUR_USERNAME--action-recognition-api-fastapi-app.modal.run/classes

# Predict action
curl -X POST \
  -F "file=@image.jpg" \
  https://YOUR_USERNAME--action-recognition-api-fastapi-app.modal.run/predict
```

## 📁 Files Structure

```
├── action_model (2).pth           # Trained CNN-LSTM model weights
├── action_recognition_modal.py    # Modal deployment code with FastAPI
├── action_recognition_ui.html     # Web UI for the API
├── modal_app.py                   # Previous modal app (reference)
├── notebook1a2894c042.ipynb       # Training notebook
└── README.md                      # This file
```

## 🎬 Supported Action Classes (UCF101)

The model recognizes 101 action classes including:
- ApplyEyeMakeup, ApplyLipstick, Archery
- Basketball, Biking, Bowling
- PlayingGuitar, PlayingPiano, PlayingViolin
- Swimming, Running, Walking
- And 89 more...

Get the full list at: `/classes` endpoint

## 🔧 Troubleshooting

### "Connection Failed" in UI
- Verify your Modal app is deployed: `modal app list`
- Check the URL is correct (no typos)
- Ensure CORS is enabled (it's already set in the code)

### Deployment Errors
- Make sure `action_model (2).pth` is in the same folder
- Run `modal token new` if authentication expired

### Slow First Request
- First request takes ~30-60 seconds (cold start - loading model + GPU)
- Subsequent requests are much faster (~2-5 seconds)

## 💰 Cost Information

Modal.com uses a pay-per-use model:
- T4 GPU: ~$0.59/hour (billed per second)
- Cold starts are free
- Idle containers auto-shutdown after ~5 minutes

## 🛑 Stop/Delete Deployment

```bash
# List apps
modal app list

# Stop the app
modal app stop action-recognition-api

# Delete the app
modal app delete action-recognition-api
```

---

**Note**: Do not modify or delete other Modal apps. This deployment creates a new separate app named `action-recognition-api`.
