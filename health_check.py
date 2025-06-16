import argparse
import requests
import json

def main():
    parser = argparse.ArgumentParser(description="Check the /health endpoint of the deployed model.")
    parser.add_argument('--endpoint', required=True, help='Base URL of the deployed model (e.g., https://api.cortex.cerebrium.ai/v4/p-e9ee5f96/mtailor-mlops)')
    parser.add_argument('--api-key', required=True, help='Cerebrium API key')
    args = parser.parse_args()

    health_url = args.endpoint.rstrip('/') + '/health'
    headers = {
        "Authorization": f"Bearer {args.api_key}"
    }

    try:
        response = requests.get(health_url, headers=headers)
        print(f"GET {health_url}")
        print(f"Status code: {response.status_code}")
        
        # Try to parse response as JSON, fallback to text if not valid JSON
        try:
            response_content = response.json()
            print(f"Response: {json.dumps(response_content)}")
        except json.JSONDecodeError:
            response_content = response.text.strip()
            print(f"Response: {response_content}")

        if response.status_code == 200 and response_content == 'OK':
            print("Health check PASSED.")
        else:
            print("Health check FAILED.")
    except Exception as e:
        print(f"Health check FAILED with error: {e}")

if __name__ == "__main__":
    main() 