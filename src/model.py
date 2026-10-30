import os
import json
import numpy as np
import torch
import torch.nn as nn
import timm

from .config import MODEL_PATH, META_PATH, DB_PATH


def build_model(num_classes, backbone="efficientnet_b0", pretrained=True):
    backbone_model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
    feat_dim = backbone_model.num_features
    model = GeoModel(backbone_model, feat_dim, num_classes)
    return model, feat_dim


class GeoModel(nn.Module):
    def __init__(self, backbone, feat_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        if f.dim() == 4:
            f = f.mean(dim=(2, 3))
        logits = self.head(f)
        return logits, f


def save_meta(meta):
    with open(META_PATH, "w") as f:
        json.dump(meta, f)


def load_meta():
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH) as f:
        return json.load(f)


def save_db(embeddings, lats, lngs, cells):
    np.savez(DB_PATH, embeddings=embeddings, lats=lats, lngs=lngs, cells=cells)


def load_db():
    if not os.path.exists(DB_PATH):
        return None
    data = np.load(DB_PATH)
    return {
        "embeddings": data["embeddings"],
        "lats": data["lats"],
        "lngs": data["lngs"],
        "cells": data["cells"],
    }
