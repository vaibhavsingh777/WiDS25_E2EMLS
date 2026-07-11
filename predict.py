import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. Define the Labels (Alphabetical order based on PlantVillage folders)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot',
    'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_leaf_smoke', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# 2. Replicate Model Architecture
class PlantVillageModel(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        # Use weights=None because we are loading our own local .pth file
        self.base_model = models.mobilenet_v2(weights=None)
        self.base_model.classifier[1] = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

# 3. Load Model and Weights
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantVillageModel()
# Path must match your local folder structure
weights_path = "outputs/models/global_plant_model.pth"

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    print("✅ Model weights loaded successfully.")
else:
    print(f"❌ Error: Could not find weights at {weights_path}")

# 4. Image Preprocessing (Must match Training exactly)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_prediction(img_path):
    try:
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            # Convert logits to probabilities
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(prob, 1)
            
        label = CLASS_NAMES[predicted.item()]
        print(f"\n--- Prediction Result ---")
        print(f"Leaf Disease: {label}")
        print(f"Confidence:   {confidence.item()*100:.2f}%")
        print(f"Class ID:     {predicted.item()}")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    path = input("Enter the filename of the leaf image in this folder: ")
    run_prediction(path)