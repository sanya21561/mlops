# Image Classification API

This project implements a FastAPI-based image classification service using an ONNX model deployed on Cerebrium server. The server accepts base64-encoded images and returns classification predictions.

## Project Structure

```
├── main.py             # FastAPI application
├── model.py            # Model loading and preprocessing
├── convert_to_onnx.py  # Script to convert PyTorch model to ONNX
├── test.py            # Unit tests for model and preprocessing
├── test_server.py     # Integration tests for API endpoints
├── Dockerfile         # Docker configuration
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Prerequisites 

- Python 3.10
- Docker (for containerized deployment)
- Cerebrium account (for deployment)

## Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server locally:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check

To check the health of your deployed model:

```bash
python health_check.py --endpoint https://api.cortex.cerebrium.ai/v4/p-e9ee5f96/mtailor \
  --api-key eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWU5ZWU1Zjk2IiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyNjM4MTM3NjAwfQ.KoFajLmQw957Byvf7qtVmJF-nKahHPZxFxq6oBV4_Xj9X_sj7fY0J5L20hblsqk7LTHLYroMAxBL1VZoNJKp8UXErN2Wetyy_MMq6nykEFqM9FV8L4nS9YT1j1q2uObUyNv7sNG-D6Tp79r_mPOnjBkvFSNIOH32UQ5Dmif5kvroGTBmLvvZs6zAUnWwzLB818zKO4GAgCe4jaJU_xisa2j81xqfYG3EMu6inqPJfK3Xg83KOscXQ3YjzzGEKQKWiGRYx9FO1490lAO7bD4ZMU-DsolKOvWcyK1YsjURAKjeqd4o5wDu76WJfCxZt-i7fwBvja1ytYdcaj9AUKNkDg
```






## Testing

Run unit tests:
```bash
python test.py
```

### Local Server Testing

To test the FastAPI server running locally (e.g., after running `uvicorn main:app --reload`):

```bash
python test_local_server.py
```

This script sends a test image (`images/n01440764_tench.jpeg`) to the local server at `http://localhost:8000/`.

Returns prediction results:
```json
{
    "class_id": 0,
    "confidence": 0.95
}
```

## Testing the Deployed Model on Cerebrium Server

To test the deployed model, use the following commands:

### Test with a Single Image

```bash
python test_server.py --api-key eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWU5ZWU1Zjk2IiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyNjM4MTM3NjAwfQ.KoFajLmQw957Byvf7qtVmJF-nKahHPZxFxq6oBV4_Xj9X_sj7fY0J5L20hblsqk7LTHLYroMAxBL1VZoNJKp8UXErN2Wetyy_MMq6nykEFqM9FV8L4nS9YT1j1q2uObUyNv7sNG-D6Tp79r_mPOnjBkvFSNIOH32UQ5Dmif5kvroGTBmLvvZs6zAUnWwzLB818zKO4GAgCe4jaJU_xisa2j81xqfYG3EMu6inqPJfK3Xg83KOscXQ3YjzzGEKQKWiGRYx9FO1490lAO7bD4ZMU-DsolKOvWcyK1YsjURAKjeqd4o5wDu76WJfCxZt-i7fwBvja1ytYdcaj9AUKNkDg \
  --endpoint https://api.cortex.cerebrium.ai/v4/p-e9ee5f96/mtailor/ \
  --image images/n01440764_tench.jpeg
```

### Test with Preset Test Cases

```bash
python test_server.py --api-key eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWU5ZWU1Zjk2IiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyNjM4MTM3NjAwfQ.KoFajLmQw957Byvf7qtVmJF-nKahHPZxFxq6oBV4_Xj9X_sj7fY0J5L20hblsqk7LTHLYroMAxBL1VZoNJKp8UXErN2Wetyy_MMq6nykEFqM9FV8L4nS9YT1j1q2uObUyNv7sNG-D6Tp79r_mPOnjBkvFSNIOH32UQ5Dmif5kvroGTBmLvvZs6zAUnWwzLB818zKO4GAgCe4jaJU_xisa2j81xqfYG3EMu6inqPJfK3Xg83KOscXQ3YjzzGEKQKWiGRYx9FO1490lAO7bD4ZMU-DsolKOvWcyK1YsjURAKjeqd4o5wDu76WJfCxZt-i7fwBvja1ytYdcaj9AUKNkDg \
  --endpoint https://api.cortex.cerebrium.ai/v4/p-e9ee5f96/mtailor/ \
  --test-mode
```

- The `--api-key` and `--endpoint` arguments are required for all remote tests.
- The `--image` argument is for single image testing.
- The `--test-mode` flag runs the preset test suite.

## Deployment to Cerebrium

1. Build the Docker image:
```bash
docker build -t image-classification-api .
```

2. Test the Docker image locally:
```bash
docker run -p 8000:8000 image-classification-api
```

3. Deploy to Cerebrium:
```bash
pip install cerebrium
cerebrium login
cerebrium deploy image-classification-api
```

## Error Handling

The API includes comprehensive error handling for:
- Invalid image data
- Preprocessing failures
- Model prediction errors
- Server errors

All errors return appropriate HTTP status codes and descriptive error messages.

## Monitoring

The service includes logging for:
- Model initialization
- Request processing
- Error conditions
- Prediction results

## Security Considerations

- Input validation for base64 images
- Error message sanitization
- Secure dependency versions
- Container security best practices

## Performance

The service is optimized for:
- Fast image preprocessing
- Efficient model inference
- Low latency responses
- Container resource efficiency