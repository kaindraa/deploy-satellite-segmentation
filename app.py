import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from utils import load_model_checkpoint,  visualize_prediction_overlay
import os, numpy as np, pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_image(index):
    path = os.path.join(image_folder, image_files[index])
    img = Image.open(path)
    st.image(img, caption=image_files[index], use_column_width=True)


@st.cache_resource
def load_model():
    return load_model_checkpoint("models/UnetVGG16_best_val_loss.pth", num_classes=4, device=device)

model = load_model()

st.title("🛰️ Indonesia Satellite Image Segmentation")
st.markdown("""
Segment satellite images into land cover classes such as vegetation, infrastructure, bare land, and water.

The segmentation model is based on the **U-Net architecture** [(Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597), which consists of a symmetric encoder-decoder structure for precise pixel-wise prediction. The encoder is adapted from **VGG16** [(Simonyan and Zisserman, 2014)](https://arxiv.org/abs/1409.1556), a deep convolutional network pretrained on the **ImageNet** dataset [(Deng et al., 2009)](https://arxiv.org/abs/1409.0575), leveraging transfer learning to capture rich visual features.

The model was trained using a custom-annotated dataset of Indonesian satellite imagery, labeled to reflect relevant land cover types across various regions.
""")


# Hide fullscreen zoom buttons
st.markdown("""
<style>
.element-container:nth-child(3) .overlayBtn {visibility: hidden;}
.element-container:nth-child(12) .overlayBtn {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Paper figure carousel
with st.container():
    st.markdown("### 📚 Showcase: Figures from Paper")

    image_folder = "paper_figures"
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if "paper_index" not in st.session_state:
        st.session_state.paper_index = 0

    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("◀", key="prev_paper"):
            st.session_state.paper_index = (st.session_state.paper_index - 1) % len(image_files)
    with col3:
        if st.button("▶", key="next_paper"):
            st.session_state.paper_index = (st.session_state.paper_index + 1) % len(image_files)
    with col2:
        show_image(st.session_state.paper_index)

palette = {
    0: [18, 243, 44],
    1: [173, 53, 253],
    2: [193, 126, 11],
    3: [20, 20, 200]
}
labels = ["Vegetation", "Infrastructure", "Land", "Water"]

# Prediction section
st.markdown("## 🔍 Predict Segmentation")
st.markdown("Select an image from examples or upload your own to generate a segmentation mask.")

option = st.radio("Choose image source", ["📁 From examples", "📤 Upload your own"])
img, img_name = None, None

if option == "📁 From examples":
    image_dir = "test_images"
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    selected_image = st.selectbox("Select image", image_files)
    if selected_image:
        img_path = os.path.join(image_dir, selected_image)
        img = Image.open(img_path).convert("RGB")
        img_name = selected_image

elif option == "📤 Upload your own":
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_name = uploaded.name

if img:
    if st.session_state.get("last_uploaded") != img_name:
        st.session_state.last_uploaded = img_name
        st.session_state.pred = None
        st.session_state.overlay = None
        st.session_state.show_overlay = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict"):
            transform = T.Compose([
                T.Resize((480, 480), interpolation=InterpolationMode.LANCZOS),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            inp = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = torch.argmax(model(inp).squeeze(), dim=0).cpu().numpy()
            st.session_state.pred = pred
            st.session_state.overlay = visualize_prediction_overlay(img, pred, palette, alpha=0.9)
            st.session_state.show_overlay = True

    with col2:
        st.checkbox("Show overlay", key="show_overlay", disabled=st.session_state.get("pred") is None)

    st.markdown("### 🎨 Class Colors")
    cols = st.columns(len(palette))
    for i, color in palette.items():
        cols[i].markdown(
            f"<div style='display:flex;align-items:center;'>"
            f"<div style='width:20px;height:20px;background-color:rgb{tuple(color)};"
            f"margin-right:8px;border:1px solid #000'></div><b>{labels[i]}</b></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 4px;'></div>", unsafe_allow_html=True)

    display = st.session_state.overlay if st.session_state.get("show_overlay") else img
    st.image(display, width=600)

    if st.session_state.get("pred") is not None:
        st.markdown("### 📊 Predicted Class Distribution")
        st.markdown("The following table shows the percentage of each land cover class predicted in the image.")
        counts = np.bincount(st.session_state.pred.flatten(), minlength=len(palette))
        total = counts.sum()
        if total > 0:
            df = pd.DataFrame({
                "Class": labels,
                "Percentage": [f"{c / total * 100:.2f}%" for c in counts]
            })
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No class detected in prediction.")
else:
    st.warning("Please upload or select an image to begin.")
