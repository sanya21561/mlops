import requests
import base64

def test_local_server(image_path: str):
    url = "http://localhost:8000/"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "image": image_b64
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print(f"Success: {response.json()}")
    else:
        print(f"Failed with status {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    test_local_server("images/n01440764_tench.jpeg")
