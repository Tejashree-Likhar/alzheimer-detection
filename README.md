# 🧠 Alzheimer's MRI Classifier

A deep learning project that classifies brain MRI scans into four Alzheimer's disease
stages using transfer learning (ResNet18), trained on the public
[Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI) dataset
from Hugging Face.

> ⚠️ **Disclaimer:** This is an educational/research project, **not a medical device**.
> It has not been clinically validated and must never be used to make real diagnostic
> or treatment decisions. Always consult a qualified healthcare professional.

🔗 **[Live Demo](https://alzheimer-detectiongit-uwrk5nlabstx54vwmv9rtf.streamlit.app/)**

## Classes

| Label | Class              |
|-------|--------------------|
| 0     | NonDemented        |
| 1     | VeryMildDemented   |
| 2     | MildDemented       |
| 3     | ModerateDemented   |

## Project structure

```
alzheimer-detection/
├── app.py                  # Streamlit web app for interactive demo/deployment
├── requirements.txt
├── src/
│   ├── data.py              # Dataset loading & preprocessing (HF -> PyTorch)
│   ├── model.py              # ResNet18-based classifier architecture
│   ├── train.py              # Training loop with early stopping
│   ├── evaluate.py           # Test-set metrics + confusion matrix
│   ├── predict.py            # Single-image inference (CLI), with optional Grad-CAM
│   ├── gradcam.py            # Grad-CAM implementation (heatmap generation + overlay)
│   └── gradcam_grid.py       # Generates a sample Grad-CAM grid across all classes
└── .github/workflows/ci.yml  # CI: build check + lint
```

## Setup

```bash
git clone https://github.com/<your-username>/alzheimer-detection.git
cd alzheimer-detection
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

The dataset downloads automatically from the Hugging Face Hub the first time you run
training or evaluation (no manual download needed, no auth required — it's a public dataset).

## Train the model

```bash
python -m src.train --epochs 15 --batch-size 32 --lr 3e-4
```

Key flags:
- `--freeze-backbone` — only trains the classifier head (faster, good for a quick baseline)
- `--output alzheimer_model.pt` — where to save the best checkpoint
- `--patience 5` — early-stopping patience in epochs

Training uses a GPU automatically if available (`torch.cuda.is_available()`), otherwise
falls back to CPU. On a free Google Colab GPU, 15 epochs takes roughly 5–10 minutes.

## Evaluate

```bash
python -m src.evaluate --checkpoint alzheimer_model.pt
```

Prints a per-class precision/recall/F1 report and saves a confusion matrix image
(`confusion_matrix.png`).

## Predict on a single image

```bash
python -m src.predict --image path/to/scan.jpg --checkpoint alzheimer_model.pt
```

Add `--gradcam` to also save a heatmap showing which regions of the scan drove the prediction:

```bash
python -m src.predict --image path/to/scan.jpg --gradcam --gradcam-output cam.png
```

## Grad-CAM: model explainability

This project uses [Grad-CAM](https://arxiv.org/abs/1610.02391) to visualize which
regions of an MRI scan most influenced the model's prediction — important for a
medical-adjacent task where "black box" predictions aren't good enough on their own.

**Generate a heatmap for one image:**
```bash
python -m src.predict --image path/to/scan.jpg --gradcam
```

**Generate a grid of sample explanations across all four classes** (good for a
README screenshot or model report):
```bash
python -m src.gradcam_grid --checkpoint alzheimer_model.pt --output gradcam_grid.png
```

**Interactively in the web app:** the Streamlit app (`app.py`) shows the Grad-CAM
overlay automatically next to every prediction, and lets you switch which class's
heatmap to view (e.g. "what would make the model think this is ModerateDemented?").

Under the hood, `src/gradcam.py` hooks into `model.layer4` (the last convolutional
block of the ResNet18 backbone) to capture activations and gradients, then produces
a class-weighted activation map that gets upsampled and blended over the original scan.

## Deploy as a web app

The included `app.py` is a Streamlit app with drag-and-drop image upload and live
predictions.

**Run locally:**
```bash
streamlit run app.py
```

**Deploy for free on Streamlit Community Cloud:**
1. Push this repo to GitHub, including your trained `alzheimer_model.pt`
   (use [Git LFS](https://git-lfs.com/) if the checkpoint is large — ResNet18 checkpoints
   are usually ~45MB, which is fine for a normal git push).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click
   "New app", select this repo, and set the main file to `app.py`.
3. Streamlit installs `requirements.txt` automatically and gives you a public URL.

**Alternative deployment options:**
- **Hugging Face Spaces** (also free, supports Streamlit/Gradio natively)
- **Docker + any cloud provider** (Render, Railway, Fly.io, AWS/GCP/Azure) if you want
  more control

## Model details

- **Backbone:** ResNet18 pretrained on ImageNet, fine-tuned end-to-end
- **Head:** Dropout → Linear(512→128) → ReLU → Dropout → Linear(128→4)
- **Input:** MRI slices converted to 3-channel, resized to 224×224, normalized with
  ImageNet statistics
- **Augmentation (train only):** random resized crop, horizontal flip, small rotations
- **Loss:** cross-entropy
- **Optimizer:** AdamW with `ReduceLROnPlateau` scheduling on validation accuracy
- **Early stopping:** on validation accuracy plateau

## Known limitations

- The dataset is a single public collection of 2D MRI slices, not a large multi-site
  clinical cohort like ADNI — expect the model to **not generalize well** to scans from
  different scanners/protocols.
- Class imbalance: `ModerateDemented` has far fewer examples than the other classes,
  so recall on that class is typically the weakest. Check the confusion matrix.
- 2D slices lose 3D spatial context that a full volumetric (3D CNN) approach would capture.

### Ideas for extending this project
- Fine-tune on [ADNI](https://adni.loni.usc.edu) or [OASIS](https://www.oasis-brains.org)
  for stronger clinical validity (requires data use agreement).
- Try a 3D CNN on full MRI volumes instead of 2D slices.
- Add Grad-CAM visualizations to show which brain regions drove each prediction.
- Address class imbalance with weighted loss or oversampling.

## License

MIT — see [LICENSE](LICENSE). The underlying dataset (Falah/Alzheimer_MRI) has its own
license on Hugging Face; check there before commercial use.
