import streamlit as st
from PIL import Image
import os
import tempfile
import shutil
import torch
import clip

st.set_page_config(page_title="AI Image Culling Tool", layout="wide")
st.title("📸 AI Image Culling Assistant")

st.write("Upload your images and let the AI help you cull them based on your taste: sharp, well-lit, composed using rule of thirds, and more.")

# Load CLIP model and preprocessing
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# Define your labels and text prompts
labels = ["keep", "maybe", "reject"]
prompts = {
    "keep": "a sharp, well-lit photo with eyes open and good composition",
    "maybe": "a decent photo with okay lighting or focus",
    "reject": "a blurry, overexposed, or underexposed image"
}

text_tokens = clip.tokenize(list(prompts.values())).to(device)

# Upload image files
uploaded_files = st.file_uploader("Upload JPEG images", type=["jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    keep, maybe, reject = [], [], []
    tmpdir = tempfile.mkdtemp()

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            similarity = (image_features @ text_features.T).softmax(dim=-1)

        probs = similarity.squeeze().tolist()
        predicted_index = probs.index(max(probs))
        label = labels[predicted_index]

        if label == "keep":
            keep.append(uploaded_file)
        elif label == "maybe":
            maybe.append(uploaded_file)
        else:
            reject.append(uploaded_file)

    # Display results
    def show_images(images, title):
        st.subheader(f"{title} ({len(images)})")
        cols = st.columns(5)
        for i, file in enumerate(images):
            with cols[i % 5]:
                st.image(file, use_column_width=True)

    show_images(keep, "✅ Keep")
    show_images(maybe, "🤔 Maybe")
    show_images(reject, "❌ Reject")

    st.success("Done culling your images!")
