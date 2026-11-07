import torch
import timm
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from mtcnn import MTCNN

# ---------------- CONFIG ----------------
MODEL_PATH = "E:/CSMSS Projects/Sem6_mini_project/deepfake-detection-XceptionNet-main/own_train/runs/20260422_231937_xception/checkpoints/best_model.pt"
IMAGE_PATH = "C:/Users/Harshal/OneDrive/Pictures/Screenshots/check.png"
IMAGE_SIZE = 299
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- INIT ----------------
detector = MTCNN()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LOAD MODEL ----------------
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = timm.create_model("legacy_xception", pretrained=False, num_classes=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model

# ---------------- FACE EXTRACTION ----------------
def extract_face(image):
    result = detector.detect_faces(image)

    if len(result) == 0:
        return None

    # Take highest confidence face
    face = max(result, key=lambda x: x["confidence"])
    x, y, w, h = face["box"]

    # Add margin (VERY IMPORTANT)
    margin = int(0.3 * w)
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, image.shape[1])
    y2 = min(y + h + margin, image.shape[0])

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    return face_crop

# ---------------- PREDICT ----------------
def predict_image(model, image_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face = extract_face(image)

    if face is None:
        print(f"No face detected in {image_path.name}")
        return "UNKNOWN", 0.0

    tensor = transform(face).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()

    pred = "FAKE" if prob >= THRESHOLD else "REAL"

    return pred, prob

# ---------------- MAIN ----------------
if __name__ == "__main__":
    model = load_model()

    path = Path(IMAGE_PATH)

    if path.is_file():
        pred, prob = predict_image(model, path)
        print(f"\nPrediction: {pred}")
        print(f"Confidence: {prob:.4f}")

    elif path.is_dir():
        print("\nTesting folder...\n")
        for img_path in path.glob("*.*"):
            pred, prob = predict_image(model, img_path)
            print(f"{img_path.name} → {pred} ({prob:.4f})")

    else:
        print("Invalid path!")