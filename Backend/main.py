import base64
import logging
import time
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo_api")

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv5 model at startup (to avoid re-loading on each request)
model = None
@app.on_event("startup")
def load_model():
    global model
    try:
        # Load custom YOLOv5 model (replace 'best.pt' with your trained model path)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/best.pt', force_reload=False)
        model.conf = 0.2  # set low confidence threshold for predictions (0.2):contentReference[oaicite:4]{index=4}
        logger.info("YOLOv5 model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load YOLOv5 model: {e}")
        # If the model fails to load, it's critical – we raise an exception.
        raise

# Pydantic model for request body (expects base64 image and optional flag)
class PredictRequest(BaseModel):
    image_base64: str
    return_image: bool = False

@app.post("/predict")
async def predict(request: PredictRequest):
    """Receive a base64-encoded image, run YOLOv5 detection, and return results."""
    # Decode the base64 image to raw bytes
    try:
        image_data = base64.b64decode(request.image_base64)
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")
    
    # Convert bytes to NumPy array and decode into an image (BGR color space by default)
    np_array = np.frombuffer(image_data, np.uint8)
    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        # cv2.imdecode returns None if the image data is invalid
        raise HTTPException(status_code=400, detail="Image decoding failed.")
    
    # Convert BGR to RGB for YOLOv5 (model expects RGB input):contentReference[oaicite:5]{index=5}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = img_rgb.shape[0], img_rgb.shape[1]
    
    # Resize image to width 640 (height scaled proportionally) to match training resolution
    target_width = 640
    # Compute target height to maintain aspect ratio
    target_height = int(orig_height * (target_width / orig_width))
    img_resized = cv2.resize(img_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    logger.info(f"Received image {orig_width}x{orig_height}px; resized to {target_width}x{target_height}px.")
    
    # Save the processed image for debugging (convert RGB back to BGR for correct colors in file)
    try:
        debug_img = Image.fromarray(img_resized)  # PIL expects RGB array
        debug_img.save("last_processed_image.jpg")
    except Exception as e:
        logger.warning(f"Could not save debug image: {e}")
    
    # Ensure model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    # Run inference with YOLOv5 model and measure time
    start_time = time.time()
    try:
        results = model(img_resized)  # perform inference on the image
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference error.")
    inference_time = time.time() - start_time
    logger.info(f"YOLOv5 inference completed in {inference_time:.3f} seconds.")
    
    # Parse model results: extract bounding boxes and labels
    detections = []
    if results and hasattr(results, "xyxy"):
        # results.xyxy[0] is a tensor of shape (N,6) for N detections: [x1, y1, x2, y2, confidence, class]
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)  # coordinates as integers
            detections.append({
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "confidence": float(conf),
                "label": results.names[int(cls)] if hasattr(results, "names") else int(cls)
            })
    else:
        # No detections found (or unexpected results format)
        detections = []
    
    # Optionally render the detections on the image and include it in the response
    annotated_image_b64 = None
    if request.return_image:
        try:
            # Draw bounding boxes and labels on the image
            results.render()  # updates results.ims with rendered results (draws boxes/labels)
            # `results.ims` is a list of numpy arrays (BGR images) corresponding to input images
            if len(results.ims) > 0:
                # Convert the annotated image array to PIL Image for encoding
                annotated_img = results.ims[0]  # numpy array (BGR) with drawings
                annotated_pil = Image.fromarray(annotated_img)
                buffered = BytesIO()
                annotated_pil.save(buffered, format="JPEG")
                annotated_image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate annotated image: {e}")
            annotated_image_b64 = None
    
    # Prepare the JSON response
    response = {
        "boxes": detections,
        "inference_time": round(inference_time, 3)
    }
    if annotated_image_b64:
        response["image_base64"] = annotated_image_b64
    
    return response
