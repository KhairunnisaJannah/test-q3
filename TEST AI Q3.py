import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
import pandas as pd

# -----------------------------
# 1. Page config
# -----------------------------
st.set_page_config(
    page_title="Live Camera Image Classification",
    layout="centered"
)

st.title("📸 Live Camera Image Classification")
st.write("Using **ResNet-18 (ImageNet)** + Streamlit")

# -----------------------------
# 2. Load labels
# -----------------------------
@st.cache_data
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    labels = response.text.strip().split("\n")
    return labels

# -----------------------------
# 3. Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model

labels = load_imagenet_labels()
model = load_model()

# -----------------------------
# 4. Preprocessing
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 5. Camera Input
# -----------------------------
camera_image = st.camera_input("Take a photo")

if camera_image is not None:
    image = Image.open(camera_image).convert("RGB")
    st.image(image, caption="Captured Image", use_container_width=True)

    # Preprocess
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = F.softmax(outputs[0], dim=0)

    # Top 5
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    st.subheader("🔍 Top-5 Predictions")

    results = []
    for i in range(top5_prob.size(0)):
        label = labels[top5_catid[i]]
        prob = top5_prob[i].item()
        st.write(f"**{label}** — {prob:.4f}")
        results.append([label, prob])

    # Table
    df = pd.DataFrame(results, columns=["Label", "Probability"])
    st.dataframe(df, use_container_width=True)

else:
    st.info("👆 Take a photo using your camera to start.")
