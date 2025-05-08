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

# Culling profile dropdown
profile = st.selectbox("Choose a culling profile", ["Portraits", "Cars", "Events"])

# Define prompts per profile
if profile == "Portraits":
    prompts = {
        "keep": "a sharp, well-lit portrait with eyes open and good composition",
        "maybe": "a decent portrait with okay lighting or focus",
        "reject": "a blurry, poorly lit portrait with closed eyes"
    }
elif profile == "Cars":
    prompts = {
        "keep": "a sharp, centered car photo with clean background",
        "maybe": "a car photo with decent lighting or some motion blur",
        "reject": "a blurry or overexposed photo of a car"
    }
else:  # Events
    prompts = {
        "keep": "a sharp, expressive event photo with good lighting",
        "maybe": "a decent event photo with okay focus",
        "reject": "a blurry or poorly lit event photo"
    }

labels = list(prompts.keys())
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

    # Manual override UI
    def move_images(source, source_label, target_label):
        all_selected = st.checkbox(f"Select all from {source_label} to move to {target_label}")
        selected = st.multiselect(
            f"Select images to move from {source_label} to {target_label}",
            source,
            default=source if all_selected else [],
            format_func=lambda x: x.name
        )
        return [img for img in source if img not in selected], selected

    st.subheader("✅ Keep")
    st.image([img for img in keep], use_container_width=True)
    keep, moved_to_keep = move_images(maybe, "Maybe", "Keep")
    keep.extend(moved_to_keep)
    keep, moved_from_reject = move_images(reject, "Reject", "Keep")
    keep.extend(moved_from_reject)

    st.subheader("🤔 Maybe")
    st.image([img for img in maybe], use_container_width=True)
    maybe, moved_to_maybe = move_images(keep, "Keep", "Maybe")
    maybe.extend(moved_to_maybe)
    maybe, moved_from_reject = move_images(reject, "Reject", "Maybe")
    maybe.extend(moved_from_reject)

    st.subheader("❌ Reject")
    st.image([img for img in reject], use_container_width=True)
    reject, moved_to_reject = move_images(keep, "Keep", "Reject")
    reject.extend(moved_to_reject)
    reject, moved_from_maybe = move_images(maybe, "Maybe", "Reject")
    reject.extend(moved_from_maybe)

    # Export to ZIP button
    from zipfile import ZipFile

    export_path = os.path.join(tmpdir, "culled_photos")
    os.makedirs(os.path.join(export_path, "keep"), exist_ok=True)
    os.makedirs(os.path.join(export_path, "maybe"), exist_ok=True)
    os.makedirs(os.path.join(export_path, "reject"), exist_ok=True)

    for img in keep:
        with open(os.path.join(export_path, "keep", img.name), "wb") as f:
            f.write(img.getbuffer())

    for img in maybe:
        with open(os.path.join(export_path, "maybe", img.name), "wb") as f:
            f.write(img.getbuffer())

    for img in reject:
        with open(os.path.join(export_path, "reject", img.name), "wb") as f:
            f.write(img.getbuffer())

    zip_path = shutil.make_archive(export_path, 'zip', export_path)
    with open(zip_path, "rb") as f:
        st.download_button("📦 Download Sorted ZIP", f, file_name="culled_photos.zip")

    st.success("✅ Culling complete! You may now download or export your results.")
