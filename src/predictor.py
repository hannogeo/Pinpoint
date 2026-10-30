import os
import json
import math
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import MODEL_PATH, META_PATH, DB_PATH
from .model import GeoModel, load_meta, load_db
from .dataset import IMAGENET_MEAN, IMAGENET_STD


def _haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    meta = load_meta()
    num_classes = ckpt["num_classes"]
    backbone = ckpt["backbone"]
    feat_dim = ckpt["feat_dim"]
    # rebuild backbone
    import timm
    backbone_model = timm.create_model(backbone, pretrained=False, num_classes=0)
    model = GeoModel(backbone_model, feat_dim, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, meta


def preprocess(img, size=224):
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return t(img).unsqueeze(0)


def predict(image, top_k=5):
    model, meta = load_model()
    if model is None:
        return None
    db = load_db()
    device = next(model.parameters()).device
    size = meta.get("img_size", 224) if meta else 224
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")
    x = preprocess(img, size=size).to(device)
    with torch.no_grad():
        logits, f = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        f = torch.nn.functional.normalize(f, dim=1)[0].cpu().numpy()
    # top cells
    top_idx = np.argsort(probs)[::-1][:top_k]
    centers = np.array(meta["centers"]) if meta else None
    top_cells = []
    for i in top_idx:
        c = centers[i] if centers is not None else None
        top_cells.append({"cell": int(i), "prob": float(probs[i]), "center": (float(c[0]), float(c[1])) if c is not None else None})
    predicted_cell = int(top_idx[0])
    predicted_center = centers[predicted_cell] if centers is not None else None

    # kNN pin: prefer training points inside the predicted cell to avoid drift
    pin_lat, pin_lng = None, None
    if db is not None:
        embs = db["embeddings"]
        sims = embs @ f
        cell_mask = db["cells"] == predicted_cell
        if cell_mask.sum() >= 3:
            cand = np.where(cell_mask)[0]
        else:
            cand = np.arange(len(sims))
        cand_sims = sims[cand]
        k = min(top_k, len(cand))
        order = np.argsort(cand_sims)[::-1][:k]
        idx = cand[order]
        w = np.clip(sims[idx], 0, None)
        if w.sum() == 0:
            w = np.ones_like(w)
        pin_lat = float(np.sum(w * db["lats"][idx]) / np.sum(w))
        pin_lng = float(np.sum(w * db["lngs"][idx]) / np.sum(w))

    # estimated error radius from spread of training points in predicted cell
    err_km = None
    if db is not None:
        cell_mask = db["cells"] == predicted_cell
        clats = db["lats"][cell_mask]
        clngs = db["lngs"][cell_mask]
        if len(clats) >= 2:
            clat = float(np.mean(clats))
            clng = float(np.mean(clngs))
            dists = []
            for la, lo in zip(clats, clngs):
                dists.append(_haversine_km(float(la), float(lo), clat, clng))
            err_km = float(np.sqrt(np.mean(np.array(dists) ** 2)))
        elif len(clats) == 1:
            err_km = 50.0
    if err_km is None:
        err_km = 200.0

    if pin_lat is None:
        if predicted_center is not None:
            pin_lat, pin_lng = float(predicted_center[0]), float(predicted_center[1])
        else:
            pin_lat, pin_lng = 0.0, 0.0

    conf = float(probs[predicted_cell])
    if conf >= 0.5:
        band = "High"
    elif conf >= 0.25:
        band = "Medium"
    else:
        band = "Low"
    return {
        "lat": pin_lat,
        "lng": pin_lng,
        "cell": predicted_cell,
        "cell_center": (float(predicted_center[0]), float(predicted_center[1])) if predicted_center is not None else None,
        "confidence": conf,
        "confidence_band": band,
        "estimated_error_km": err_km,
        "top_cells": top_cells,
    }
