"""
Run inference on a single MRI image.

Usage:
    python -m src.predict --image path/to/scan.jpg --checkpoint alzheimer_model.pt

    # Also save a Grad-CAM heatmap showing which regions drove the prediction:
    python -m src.predict --image path/to/scan.jpg --gradcam --gradcam-output cam.png
"""

import argparse

import torch
from PIL import Image

from src.data import get_transforms, CLASS_NAMES
from src.model import build_model
from src.gradcam import GradCAM, overlay_heatmap_on_image


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    class_names = CLASS_NAMES
    return model, class_names


def predict_image(model, class_names, image_path: str, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = probs.argmax()
    return class_names[pred_idx], {name: float(prob) for name, prob in zip(class_names, probs)}


def main():
    parser = argparse.ArgumentParser(description="Predict Alzheimer's stage from an MRI image")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="alzheimer_model.pt")
    parser.add_argument("--gradcam", action="store_true",
                         help="Also generate a Grad-CAM heatmap for the prediction")
    parser.add_argument("--gradcam-output", type=str, default="gradcam_overlay.png",
                         help="Where to save the Grad-CAM overlay image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.checkpoint, device)
    prediction, probabilities = predict_image(model, class_names, args.image, device)

    print(f"\nPredicted class: {prediction}\n")
    print("Class probabilities:")
    for name, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        print(f"  {name:<20s} {prob*100:5.2f}%")

    if args.gradcam:
        original_image = Image.open(args.image).convert("RGB")
        transform = get_transforms(train=False)
        input_tensor = transform(original_image).unsqueeze(0).to(device)

        cam = GradCAM(model, target_layer=model.layer4)
        heatmap, class_idx, _ = cam.generate(input_tensor)
        cam.remove_hooks()

        overlay = overlay_heatmap_on_image(heatmap, original_image)
        overlay.save(args.gradcam_output)
        print(f"\nGrad-CAM overlay for class '{class_names[class_idx]}' saved to {args.gradcam_output}")


if __name__ == "__main__":
    main()
