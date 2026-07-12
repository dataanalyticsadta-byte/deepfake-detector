"""
Quick local test helper for the detector.

Usage:
  python scripts/predict_single.py path/to/image.jpg
"""
import sys
from PIL import Image
from detector import predict_pil, CLASSIFIER_TYPE, CLASSIFIER_PATH

if len(sys.argv) != 2:
    print('Usage: python scripts/predict_single.py path/to/image.jpg')
    sys.exit(1)

path = sys.argv[1]
img = Image.open(path).convert('RGB')
result = predict_pil(img)
print('MODEL LOADED:', CLASSIFIER_TYPE, CLASSIFIER_PATH)
print('RESULT:')
print(result)
