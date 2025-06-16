import base64
import logging
from io import BytesIO
from typing import Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from model import ImagePreprocessor, OnnxModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="API for classifying images using ONNX model",
    version="1.0.0"
)

# Initialize model and preprocessor
try:
    model = OnnxModelLoader("model.onnx")
    preprocessor = ImagePreprocessor()
    logger.info("Model and preprocessor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise


class ImageRequest(BaseModel):
    """Request model for image classification."""
    image: str  # Base64 encoded image


@app.get("/")
async def root():
    """Welcome message and documentation."""
    return {
        "message": "Welcome to the Image Classification API",
        "usage": {
            "endpoint": "POST /",
            "request_format": {
                "image": "base64_encoded_image_string"
            },
            "response_format": {
                "class_id": "predicted_class_id",
                "confidence": "prediction_confidence"
            }
        }
    }


@app.post("/")
async def predict(request: ImageRequest) -> Dict[str, Any]:
    """
    Predict class for input image.
    
    Args:
        request: ImageRequest containing base64 encoded image
        
    Returns:
        Dict containing predicted class_id and confidence
        
    Raises:
        HTTPException: If image processing or prediction fails
    """
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image data: {str(e)}"
            )
            
        # Preprocess image
        try:
            input_array = preprocessor.preprocess(image)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Image preprocessing failed: {str(e)}"
            )
            
        # Get prediction
        try:
            class_id = model.predict(input_array)
            
            # Get confidence score
            outputs = model.session.run(
                [model.output_name],
                {model.input_name: input_array}
            )
            confidence = float(np.max(outputs[0]))
            
            return {
                "class_id": class_id,
                "confidence": confidence
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.get("/health")
def health():
    return "OK"

