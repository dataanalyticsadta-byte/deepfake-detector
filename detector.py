import torch
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained EfficientNet
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

def predict(image_path):

    image = Image.open(image_path).convert("RGB")
    return predict_pil(image)


def predict_pil(image):
    """Run prediction on a PIL.Image and return the same dict as `predict`.

    This avoids writing frames to disk for realtime use.
    """
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    score = torch.softmax(output, dim=1).max().item()

    if score > 0.80:
        return {
            "prediction": "Looks Real",
            "confidence": round(score * 100, 2)
        }
    else:
        return {
            "prediction": "Potential Fake",
            "confidence": round((1 - score) * 100, 2)
        }