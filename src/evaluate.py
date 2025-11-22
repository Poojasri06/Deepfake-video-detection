# src/evaluate.py
import yaml
import torch
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from src.dataset import make_dataloaders
from src.model import DeepfakeModel  # Replace with the exact class you used to train if needed
from tqdm import tqdm
import numpy as np

def evaluate(cfg_path="configs/config.yaml", ckpt_path=None):
    """
    Evaluate the trained Deepfake detection model.

    Args:
        cfg_path (str): Path to the YAML config file.
        ckpt_path (str): Path to the trained model checkpoint (.pth).
    """
    # -----------------------
    # Load Config
    # -----------------------
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    tcfg = cfg['train']

    # -----------------------
    # Dataloaders
    # -----------------------
    _, val_loader = make_dataloaders(
        real_dir=data_cfg['real_dir'],
        fake_dir=data_cfg['fake_dir'],
        img_size=tcfg['img_size'],
        batch_size=tcfg['batch_size'],
        num_workers=tcfg['num_workers']
    )

    # -----------------------
    # Device
    # -----------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -----------------------
    # Model
    # -----------------------
    model = DeepfakeModel(model_name=tcfg['model_name'], pretrained=False).to(device)
    
    if ckpt_path:
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint

        # Load state dict safely
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print("Warning: Missing keys in checkpoint:", missing_keys)
        if unexpected_keys:
            print("Warning: Unexpected keys in checkpoint:", unexpected_keys)
        print("Checkpoint loaded successfully.")

    model.eval()

    # -----------------------
    # Evaluation
    # -----------------------
    y_true, y_pred, y_probs = [], [], []

    print("\nEvaluating...")
    for imgs, labels in tqdm(val_loader, desc="Validation"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        y_pred.extend(preds.tolist())
        y_probs.extend(probs.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # -----------------------
    # Metrics
    # -----------------------
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_true, y_probs)
        print(f"ROC AUC: {auc:.4f}")
    except Exception as e:
        print("ROC AUC calculation failed:", e)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate(ckpt_path="D:/projects/DEEPFAKE_DETECTION/output/model_video.pth")
