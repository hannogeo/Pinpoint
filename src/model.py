import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .config import MODEL_PATH, META_PATH, DB_PATH


def build_model(num_classes, backbone="efficientnet_b0", pretrained=True,
                emb_dim=512, head_type="arcface", scale=30.0):
    backbone_model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
    feat_dim = backbone_model.num_features
    model = GeoModel(backbone_model, feat_dim, num_classes,
                     emb_dim=emb_dim, head_type=head_type, scale=scale)
    return model, feat_dim


class GeoModel(nn.Module):
    """Backbone + metric-learning head.

    ``head_type="arcface"`` projects the backbone features into a compact
    embedding and classifies with a normalized (angular) classifier.  The
    projection is what makes kNN retrieval over the stored embeddings work:
    plain ImageNet features put visually similar images next to each other,
    but not necessarily *geographically* similar ones.

    ``head_type="linear"`` is the legacy layout (logits straight from the
    backbone features) so older checkpoints keep working.
    """

    def __init__(self, backbone, feat_dim, num_classes,
                 emb_dim=512, head_type="arcface", scale=30.0):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.head_type = head_type
        self.scale = float(scale)
        if head_type == "arcface":
            self.emb_dim = int(emb_dim)
            self.proj = nn.Linear(feat_dim, self.emb_dim, bias=False)
            self.weight = nn.Parameter(torch.empty(num_classes, self.emb_dim))
            nn.init.xavier_uniform_(self.weight)
        else:
            self.emb_dim = self.feat_dim
            self.head = nn.Linear(feat_dim, num_classes)

    def features(self, x):
        f = self.backbone(x)
        if f.dim() == 4:
            f = f.mean(dim=(2, 3))
        return f

    def embed(self, x):
        return self.embed_features(self.features(x))

    def embed_features(self, f):
        """Map already-extracted backbone features into the retrieval space."""
        z = self.proj(f) if self.head_type == "arcface" else f
        return F.normalize(z, dim=1)

    def logits_from_emb(self, z):
        if self.head_type == "arcface":
            return self.scale * (z @ F.normalize(self.weight, dim=1).t())
        return self.head(z)

    def forward(self, x):
        if self.head_type == "arcface":
            z = self.embed(x)
            return self.logits_from_emb(z), z
        f = self.features(x)
        return self.head(f), F.normalize(f, dim=1)


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
