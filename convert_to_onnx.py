import torch
from PIL import Image
from pytorch_model import Classifier, BasicBlock

def convert_model_to_onnx(weights_path: str, image_path: str, output_path: str = "model.onnx"):
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    input_tensor = model.preprocess_numpy(img)  # [3, 224, 224]
    input_tensor = input_tensor.unsqueeze(0)    # [1, 3, 224, 224]

    torch.onnx.export(
        model,
        input_tensor,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=11,
        do_constant_folding=True
    )

    print(f"ONNX model exported successfully to: {output_path}")

if __name__ == "__main__":
    weights_file = "weights/pytorch_model_weights.pth"
    image_file = "images/n01440764_tench.jpeg"
    convert_model_to_onnx(weights_file, image_file)
