import argparse
import json
import logging
import time
from pathlib import Path
import requests
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.WARNING)  
logger = logging.getLogger(__name__)

class CerebriumTester:
    """Test class for Cerebrium deployed model."""
    
    def __init__(self, api_key: str, endpoint_url: str):
        """
        Initialize Cerebrium tester.
        
        Args:
            api_key (str): Cerebrium API key
            endpoint_url (str): Model endpoint URL
        """
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        import base64
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
            
    def predict(self, image_path: str) -> int:
        """
        Get prediction from deployed model.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            int: predicted_class
            
        Raises:
            RuntimeError: If prediction fails
        """
        try:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            # Prepare request
            image_data = self._encode_image(image_path)
            payload = {"image": image_data}
            
            # Make request
            response = requests.post(
                self.endpoint_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            return result["class_id"]
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Failed to get prediction: {str(e)}")
            
    def run_tests(self, test_cases: List[Dict[str, any]]) -> None:
        """
        Run a series of test cases.
        
        Args:
            test_cases: List of dicts with 'image_path' and 'expected_class'
        """
        total_tests = len(test_cases)
        passed_tests = 0
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            image_path = test_case["image_path"]
            expected_class = test_case["expected_class"]
            try:
                predicted_class = self.predict(image_path)
                if predicted_class == expected_class:
                    print(f"Image: {image_path} | Predicted: {predicted_class} | Expected: {expected_class} | PASS")
                    passed_tests += 1
                    results.append((image_path, expected_class, predicted_class, True))
                else:
                    print(f"Image: {image_path} | Predicted: {predicted_class} | Expected: {expected_class} | FAIL")
                    results.append((image_path, expected_class, predicted_class, False))
            except Exception as e:
                print(f"Image: {image_path} | Error: {str(e)} | FAIL")
                results.append((image_path, expected_class, None, False))
        
        print("\nTest Summary:")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        if passed_tests == total_tests:
            print("All tests passed!")
        else:
            print(f"{total_tests - passed_tests} tests failed.")

def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Test Cerebrium deployed model")
    parser.add_argument("--api-key", required=True, help="Cerebrium API key")
    parser.add_argument("--endpoint", required=True, help="Model endpoint URL")
    parser.add_argument("--test-mode", action="store_true", help="Run preset test cases")
    parser.add_argument("--image", help="Path to single test image")
    args = parser.parse_args()
    
    tester = CerebriumTester(args.api_key, args.endpoint)
    
    if args.test_mode:
        # Preset test cases
        test_cases = [
            {
                "image_path": "images/n01440764_tench.jpeg",
                "expected_class": 0
            },
            {
                "image_path": "images/n01667114_mud_turtle.JPEG",
                "expected_class": 35
            }
        ]
        tester.run_tests(test_cases)
        
    elif args.image:
        # Single image test
        try:
            predicted_class = tester.predict(args.image)
            print(f"Image: {args.image} | Predicted: {predicted_class}")
        except Exception as e:
            print(f"Test failed: {str(e)}")
            
    else:
        parser.error("Either --test-mode or --image must be specified")


if __name__ == "__main__":
    main()


