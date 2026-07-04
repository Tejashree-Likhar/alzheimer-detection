"""
Streamlit web app for interactive Alzheimer's MRI classification.

Run locally:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud:
    1. Push this repo to GitHub (include your trained alzheimer_model.pt,
       or use Git LFS / cloud storage if it's large).
    2. Go to https://share.streamlit.io, connect your GitHub repo, and point
       it at app.py. Done — no server config needed.
"""

import streamlit as st
import torch
from PIL import Image

from src.data import get_transforms, CLASS_NAMES
from src.model import build_model
from src.gradcam import GradCAM, overlay_heatmap_on_image

st.set_page_config(page_title="Alzheimer's MRI Classifier", page_icon="🧠", layout="centered")

CHECKPOINT_PATH = "alzheimer_model.pt"


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    class_names = CLASS_NAMES
    return model, class_names, device


st.title("🧠 Alzheimer's MRI Classifier")
st.write(
    "Upload a brain MRI slice to classify it into one of four stages: "
    "**NonDemented**, **VeryMildDemented**, **MildDemented**, or **ModerateDemented**."
)

st.warning(
    "⚠️ **This is a research/educational demo, not a diagnostic tool.** "
    "It is trained on a limited public dataset and must not be used for real "
    "clinical decisions. Always consult a qualified medical professional."
)

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI scan", use_container_width=True)

    try:
        model, class_names, device = load_model()
    except FileNotFoundError:
        st.error(
            f"No model checkpoint found at `{CHECKPOINT_PATH}`. "
            "Train one first with `python -m src.train`, or download a pretrained "
            "checkpoint and place it in the repo root."
        )
        st.stop()

    transform = get_transforms(train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = probs.argmax()
    st.subheader(f"Prediction: **{class_names[pred_idx]}**")

    st.write("Class probabilities:")
    for name, prob in sorted(zip(class_names, probs), key=lambda x: -x[1]):
        st.progress(float(prob), text=f"{name}: {prob*100:.2f}%")

    st.divider()
    st.subheader("🔍 Grad-CAM: what the model is looking at")
    st.caption(
        "The highlighted regions show which parts of the scan most influenced the "
        "prediction above. Warmer colors (red/yellow) = higher influence."
    )

    show_class = st.selectbox(
        "Explain prediction for class:",
        options=class_names,
        index=int(pred_idx),
        help="By default shows the heatmap for the predicted class, but you can "
             "inspect what the model looks for in any class.",
    )
    explain_idx = class_names.index(show_class)

    # Grad-CAM needs gradients, so this runs outside the no_grad() block above
    cam = GradCAM(model, target_layer=model.layer4)
    heatmap, _, _ = cam.generate(tensor, class_idx=explain_idx)
    cam.remove_hooks()

    overlay = overlay_heatmap_on_image(heatmap, image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(overlay, caption=f"Grad-CAM: {show_class}", use_container_width=True)
