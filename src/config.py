import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MODELS_DIR = os.path.join(ROOT, "models")
LOCATIONS_PATH = os.path.join(DATA_DIR, "locations.json")
SAMPLES_PATH = os.path.join(DATA_DIR, "samples.json")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pt")
META_PATH = os.path.join(MODELS_DIR, "meta.json")
DB_PATH = os.path.join(MODELS_DIR, "db.npz")

for d in (DATA_DIR, IMAGES_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)
