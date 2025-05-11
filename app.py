import streamlit as st
from PIL import Image
import os
import tempfile
import shutil
import torch
import clip
import csv
import numpy as np

st.set_page_config(page_title="AI Image Culling Tool", layout="wide", initial_sidebar_state="auto")

# Apply custom CSS for dark professional theme
st.markdown("""
<style>
    body {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #1e1e1e;
    }
    .css-1d391kg, .st-bb, .st-at, .st-ax {
        background-color: #121212;
        color: #c9d1d9;
    }
    .css-ffhzg2 {
        color: #c9d1d9;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #90caf9;
    }
    .stTextInput>div>div>input, .stTextInput>div>div>textarea {
        background-color: #1f1f1f;
        color: #ffffff;
    }
    .stDownloadButton button {
        background-color: #2c3e50;
        color: white;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI Image Culling Assistant")

st.write("Upload your images and let the AI help you cull them based on focus, lighting, and composition.")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

profile = st.selectbox("Choose a culling profile", ["Portraits", "Cars", "Events"])

use_custom_prompt = st.checkbox("Use a custom prompt instead of predefined profile prompts")
custom_prompt = ""
if use_custom_prompt:
    custom_prompt = st.text_input("Enter your custom 'keep' prompt:")

if profile == "Portraits":
    prompts = {
        "keep": "a sharp portrait with eyes open and good composition",
        "maybe": "a decent portrait with okay lighting or focus",
        "reject": "a blurry or poorly lit portrait with closed eyes"
    }
elif profile == "Cars":
    prompts = {
        "keep": "a sharp, centered photo of a car",
        "maybe": "a car photo with decent lighting or some motion blur",
        "reject": "a blurry or overexposed car photo"
    }
else:
    prompts = {
        "keep": "a sharp, expressive event photo with good lighting",
        "maybe": "a decent event photo with okay focus",
        "reject": "a blurry or poorly lit event photo"
    }

labels = list(prompts.keys())
if use_custom_prompt and custom_prompt:
    prompts = {
        "keep": custom_prompt,
        "maybe": "an average version of: " + custom_prompt,
        "reject": "a poor version of: " + custom_prompt
    }

text_tokens = clip.tokenize(list(prompts.values())).to(device)
uploaded_files = st.file_uploader("Upload JPEG images", type=["jpg", "jpeg"], accept_multiple_files=True)

manual_overrides = {}
original_labels = {}

if uploaded_files:
    keep, maybe, reject = [], [], []
    training_data = []
    embedding_data = []
    lightroom_export_data = []
    corrections_log = []
    seen_filenames = set()
    tmpdir = tempfile.mkdtemp()

    for uploaded_file in uploaded_files:
        if uploaded_file.name in seen_filenames:
            continue
        seen_filenames.add(uploaded_file.name)

        image = Image.open(uploaded_file).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            similarity = (image_features @ text_features.T).softmax(dim=-1)

        probs = similarity.squeeze().tolist()
        embedding = image_features.squeeze().cpu().numpy()
        predicted_index = probs.index(max(probs))
        label = labels[predicted_index]
        original_labels[uploaded_file.name] = label

        training_data.append([uploaded_file.name, label, profile] + probs)
        embedding_data.append([uploaded_file.name, label, profile] + embedding.tolist())
        lightroom_export_data.append([uploaded_file.name, label])
        corrections_log.append([uploaded_file.name, label, profile, probs[0], probs[1], probs[2], label])

        if label == "keep":
            keep.append((uploaded_file.name, uploaded_file, image.copy(), label))
        elif label == "maybe":
            maybe.append((uploaded_file.name, uploaded_file, image.copy(), label))
        else:
            reject.append((uploaded_file.name, uploaded_file, image.copy(), label))

    def show_thumbnails(images, title):
        st.subheader(f"{title} ({len(images)})")
        cols = st.columns(5)
        for i, (fname, file, img, orig_label) in enumerate(images):
            thumbnail = img.copy()
            thumbnail.thumbnail((200, 200))
            with cols[i % 5]:
                st.image(thumbnail, caption=fname)
                new_label = st.selectbox(f"Label for {fname}", labels, index=labels.index(orig_label), key=f"select_{title}_{fname}")
                if new_label != orig_label:
                    manual_overrides[fname] = new_label

    show_thumbnails(keep, "Keep")
    show_thumbnails(maybe, "Maybe")
    show_thumbnails(reject, "Reject")

    for fname, new_label in manual_overrides.items():
        for row in lightroom_export_data:
            if row[0] == fname:
                row[1] = new_label
        for row in corrections_log:
            if row[0] == fname:
                row.append(new_label)

    corrections_path = os.path.join(tmpdir, "corrections_log.csv")
    with open(corrections_path, "w", newline="") as correction_file:
        writer = csv.writer(correction_file)
        writer.writerow(["filename", "predicted_label", "profile", "prob_keep", "prob_maybe", "prob_reject", "final_label"])
        for row in corrections_log:
            writer.writerow(row)

    with open(corrections_path, "rb") as f:
        st.download_button("Download Corrections Log", f, file_name="corrections_log.csv")

    st.success("Culling complete. You may now download your sorted images, training data, and corrections.")
