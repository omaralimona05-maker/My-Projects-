import os
import time
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
)


class FFNN(nn.Module):
    def __init__(self, num_classes):
        super(FFNN, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(128 * 128 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



transform_inference = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading model…")
def load_model(model_class, num_classes, checkpoint_path):
    model = model_class(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    # Support both bare state-dict and checkpoint dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def predict(model, img_tensor):
    """Run inference; return predicted class index, confidence array, elapsed ms."""
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(img_tensor)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs, elapsed_ms


def confidence_bar(ax, class_names, probs, pred_idx, title):
    colors = ["#4CAF50" if i == pred_idx else "#90CAF9" for i in range(len(class_names))]
    bars = ax.barh(class_names, probs * 100, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.set_title(title, fontweight="bold")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=9)
    green_patch = mpatches.Patch(color="#4CAF50", label="Predicted class")
    ax.legend(handles=[green_patch], loc="lower right", fontsize=8)



st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown("### Class Names")
st.sidebar.caption("Enter one class per line (same order as training folders).")
default_classes = "glioma\nmeningioma\nnotumor\npituitary"
raw_classes = st.sidebar.text_area("Classes", value=default_classes, height=110)
class_names = [c.strip() for c in raw_classes.strip().splitlines() if c.strip()]
num_classes = len(class_names)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Checkpoints")

ffnn_path = st.sidebar.text_input(
    "FFNN checkpoint (.pth)",
    value="ffnn_model.pth",
    help="Path relative to this script, or absolute path.",
)
cnn_path = st.sidebar.text_input(
    "CNN checkpoint (.pth)",
    value="cnn_model.pth",
    help="Path relative to this script, or absolute path.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Select Model(s)")
run_ffnn = st.sidebar.checkbox("Run FFNN", value=True)
run_cnn  = st.sidebar.checkbox("Run CNN",  value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Device:** `{DEVICE}`  \n"
    f"**Classes ({num_classes}):** {', '.join(class_names)}"
)


st.title("🧠 Brain Tumor Image Classifier")
st.markdown(
    "Upload an MRI scan and get predictions from **FFNN** and/or **CNN** — "
    "the exact architectures trained in the notebook, loaded from your saved checkpoints."
)


ffnn_model, cnn_model = None, None
model_errors = []

if run_ffnn:
    if os.path.isfile(ffnn_path):
        try:
            ffnn_model = load_model(FFNN, num_classes, ffnn_path)
        except Exception as e:
            model_errors.append(f"FFNN load error: {e}")
    else:
        model_errors.append(f"FFNN checkpoint not found: `{ffnn_path}`")

if run_cnn:
    if os.path.isfile(cnn_path):
        try:
            cnn_model = load_model(CNN, num_classes, cnn_path)
        except Exception as e:
            model_errors.append(f"CNN load error: {e}")
    else:
        model_errors.append(f"CNN checkpoint not found: `{cnn_path}`")

for err in model_errors:
    st.warning(err)


st.markdown("---")
uploaded = st.file_uploader(
    "Upload an MRI image (JPG / PNG / BMP)",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    img_tensor = transform_inference(pil_img)

    col_img, col_results = st.columns([1, 2], gap="large")

    with col_img:
        st.subheader("Uploaded Image")
        st.image(pil_img, use_container_width=True)
        st.caption(f"Original size: {pil_img.size[0]} × {pil_img.size[1]} px → resized to 128 × 128")

    with col_results:
        st.subheader("Predictions")

        any_model_loaded = (ffnn_model is not None) or (cnn_model is not None)

        if not any_model_loaded:
            st.error("No models loaded. Check the checkpoint paths in the sidebar.")
        else:
            results = {}

            if ffnn_model is not None:
                pred_idx, probs, elapsed = predict(ffnn_model, img_tensor)
                results["FFNN"] = {
                    "pred_idx": pred_idx,
                    "probs": probs,
                    "elapsed_ms": elapsed,
                    "label": class_names[pred_idx],
                    "conf": probs[pred_idx],
                }

            if cnn_model is not None:
                pred_idx, probs, elapsed = predict(cnn_model, img_tensor)
                results["CNN"] = {
                    "pred_idx": pred_idx,
                    "probs": probs,
                    "elapsed_ms": elapsed,
                    "label": class_names[pred_idx],
                    "conf": probs[pred_idx],
                }

            # ── Result cards ──
            card_cols = st.columns(len(results))
            for col, (model_name, res) in zip(card_cols, results.items()):
                with col:
                    st.metric(
                        label=f"{model_name} Prediction",
                        value=res["label"].upper(),
                        delta=f"{res['conf']*100:.1f}% confidence",
                    )
                    st.caption(f"Inference: {res['elapsed_ms']:.1f} ms")

            # ── Confidence bar charts ──
            st.markdown("#### Confidence Breakdown")
            n_charts = len(results)
            fig, axes = plt.subplots(1, n_charts, figsize=(6 * n_charts, max(3, num_classes * 0.7)))
            if n_charts == 1:
                axes = [axes]
            for ax, (model_name, res) in zip(axes, results.items()):
                confidence_bar(ax, class_names, res["probs"], res["pred_idx"], model_name)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── Agreement indicator (only when both models run) ──
            if len(results) == 2:
                labels = [r["label"] for r in results.values()]
                if labels[0] == labels[1]:
                    st.success(f"✅ Both models agree: **{labels[0].upper()}**")
                else:
                    st.warning(
                        f"⚠️ Models disagree — "
                        f"FFNN says **{labels[0].upper()}**, "
                        f"CNN says **{labels[1].upper()}**"
                    )


st.markdown("---")
st.caption(
    "Models must be saved from the training notebook with `torch.save(model.state_dict(), 'ffnn_model.pth')` "
    "and `torch.save(cnn_model.state_dict(), 'cnn_model.pth')`. "
    "Checkpoint paths are configurable in the sidebar."
)