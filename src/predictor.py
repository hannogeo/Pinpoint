import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .config import MODEL_PATH
from .model import GeoModel, load_meta, load_db
from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .uncertainty import (haversine_km, spatial_stats, mass_within,
                          mask_cell_probs, EXACT_MATCH_SIM)


def _haversine_km(lat1, lng1, lat2, lng2):
    return haversine_km(lat1, lng1, lat2, lng2)


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    meta = load_meta()
    num_classes = ckpt["num_classes"]
    backbone = ckpt["backbone"]
    feat_dim = ckpt["feat_dim"]
    head_type = ckpt.get("head_type", "linear")
    emb_dim = ckpt.get("emb_dim", 512)
    scale = ckpt.get("arcface_scale", 30.0)
    # rebuild backbone
    import timm
    backbone_model = timm.create_model(backbone, pretrained=False, num_classes=0)
    model = GeoModel(backbone_model, feat_dim, num_classes,
                     emb_dim=emb_dim, head_type=head_type, scale=scale)
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


def _cell_spread_km(db, cell, cell_mask):
    """Legacy fallback: RMS spread of the training points inside one cell."""
    clats = db["lats"][cell_mask]
    clngs = db["lngs"][cell_mask]
    if len(clats) >= 2:
        clat = float(np.mean(clats))
        clng = float(np.mean(clngs))
        dists = [haversine_km(float(la), float(lo), clat, clng)
                 for la, lo in zip(clats, clngs)]
        return float(np.sqrt(np.mean(np.array(dists) ** 2)))
    if len(clats) == 1:
        return 50.0
    return 200.0


def _calibrated_confidence(mass, meta):
    """Map retrieval mass to the hit rate actually observed during training."""
    cmap = (meta or {}).get("confidence_map")
    if not cmap:
        return float(np.clip(mass, 0.0, 1.0)), False
    x = np.asarray(cmap.get("x", []), dtype=np.float64)
    y = np.asarray(cmap.get("y", []), dtype=np.float64)
    if len(x) < 2 or len(x) != len(y):
        return float(np.clip(mass, 0.0, 1.0)), False
    keep = np.concatenate(([True], np.diff(x) > 0))
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return float(np.clip(mass, 0.0, 1.0)), False
    val = float(np.interp(mass, x, y))
    return float(np.clip(val, 0.0, 1.0)), True


def predict(image, top_k=5, tta=True):
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
        logits, z = model(x)
        z = F.normalize(z, dim=1)
        if tta:
            # Horizontal flip: averages away camera/framing noise and makes
            # the predicted distribution a little less trigger-happy.
            logits2, z2 = model(torch.flip(x, dims=[3]))
            logits = 0.5 * (logits + logits2)
            z = F.normalize(0.5 * (z + F.normalize(z2, dim=1)), dim=1)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        f = z[0].cpu().numpy().astype(np.float32)

    # top cells (kept for the UI / debugging)
    top_idx = np.argsort(probs)[::-1][:top_k]
    centers = np.array(meta["centers"]) if meta else None
    top_cells = []
    for i in top_idx:
        c = centers[i] if centers is not None else None
        top_cells.append({
            "cell": int(i),
            "prob": float(probs[i]),
            "center": (float(c[0]), float(c[1])) if c is not None else None,
        })
    predicted_cell = int(top_idx[0])
    predicted_center = centers[predicted_cell] if centers is not None else None

    # Calibrated during training: the raw r80 has to be multiplied by this
    # factor for ~80% of held-out panoramas to actually fall inside it.
    radius_scale = float(meta.get("radius_scale", 1.0)) if meta else 1.0
    radius_scale = float(np.clip(radius_scale, 0.1, 50.0))

    pin_lat = pin_lng = None
    stats = None
    ellipse = None
    exact = False
    p25 = p100 = p500 = None

    if db is not None and len(db.get("embeddings", [])) > 0:
        sims = db["embeddings"] @ f
        exact = float(sims.max()) >= EXACT_MATCH_SIM
        cell_probs = probs if exact else mask_cell_probs(probs)
        stats = spatial_stats(
            db["embeddings"], db["lats"], db["lngs"], db.get("cells"),
            cell_probs, sims=sims,
        )
        if stats is not None:
            pin_lat, pin_lng = stats["lat"], stats["lng"]
            # Raw retrieval mass, used for display and for the calibration map.
            p25 = mass_within(stats, 25.0)
            p100 = mass_within(stats, 100.0)
            p500 = mass_within(stats, 500.0)
            e = stats.get("ellipse")
            if e is not None:
                ellipse = {
                    "major_km": float(e["major_km"] * radius_scale),
                    "minor_km": float(e["minor_km"] * radius_scale),
                    "bearing_deg": float(e["bearing_deg"]),
                }

    if pin_lat is None:
        # No embedding DB: fall back to the cell centre.
        if predicted_center is not None:
            pin_lat, pin_lng = float(predicted_center[0]), float(predicted_center[1])
        else:
            pin_lat, pin_lng = 0.0, 0.0

    conf_calibrated = False
    if stats is not None:
        err_km = float(stats["r80_km"] * radius_scale)
        conf, conf_calibrated = _calibrated_confidence(p100, meta)
        if exact:
            # A near-duplicate already in the database is essentially certain.
            conf = max(conf, 0.98)
            conf_calibrated = True
    else:
        # No retrieval at all: fall back to the spread of the predicted cell.
        if db is not None and "cells" in db:
            cell_mask = db["cells"] == predicted_cell
            err_km = _cell_spread_km(db, predicted_cell, cell_mask) * radius_scale
        else:
            err_km = 200.0 * radius_scale
        conf = float(probs[predicted_cell])

    # Bands are about how tight the pin is, not how sure the classifier is.
    if err_km <= 25:
        band = "High"
    elif err_km <= 150:
        band = "Medium"
    else:
        band = "Low"

    return {
        "lat": float(pin_lat),
        "lng": float(pin_lng),
        "cell": predicted_cell,
        "cell_center": (float(predicted_center[0]), float(predicted_center[1])) if predicted_center is not None else None,
        "confidence": float(conf),
        "confidence_band": band,
        "confidence_calibrated": bool(conf_calibrated),
        "estimated_error_km": err_km,
        "radius_80_km": err_km,
        "p_within_25km": p25,
        "p_within_100km": p100,
        "p_within_500km": p500,
        "region_confidence": float(probs[predicted_cell]),
        "ellipse": ellipse,
        "n_candidates": int(stats["n_candidates"]) if stats is not None else 0,
        "exact_match": bool(exact),
        "radius_scale": radius_scale,
        "radius_calibrated": bool(meta and meta.get("radius_scale") is not None),
        "top_cells": top_cells,
    }
