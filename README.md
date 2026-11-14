# Pinpoint

A Streamlit app that trains a region-guessing model from Google Street View panoramas
identified by `panoId`, then guesses the location of an uploaded image and pins it on an OSM map.

## Setup

```bash
git clone https://github.com/hannogeo/Pinpoint.git
cd Pinpoint
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```
or double-click `run.bat`.

## Workflow

1. **Training tab**: upload one or more JSON files (both the original format and the variant with
   `country`/`region`/`imageDate`/etc. are supported). Locations are merged and deduplicated by `panoId`.
2. Click **Train model**. The app automatically downloads panoramas by `panoId`, renders perspective
   views, and trains the model. Progress, elapsed time, and ETA are shown.
3. **Locate tab**: upload an image, click **Predict location**. Shows coordinates (copyable),
   confidence (estimated chance the true location is within 100 km), the 80% error radius, and an
   interactive OSM map with the pin and radius.

## How it works

- A frozen ImageNet backbone (timm) extracts one feature vector per rendered view. Features are
  cached in `data/features.npy`, so retraining only touches the small head.
- A projection head is trained with a CosFace/ArcFace margin, which pulls views of nearby places
  together. That projected embedding (not the raw backbone features) powers retrieval.
- Prediction runs kNN over the whole embedding database, restricted to the most likely geocells but
  still open to near-exact matches. The pin is the weighted centre of the strongest matches.
- Uncertainty comes from the actual spread of those matches: the 80% radius is a weighted quantile
  of real great-circle distances, and the radius and confidence are calibrated on
  leave-one-panorama-out predictions during training.

## Notes

- Panoramas are fetched without the Google API using the `streetlevel` library.
- Training data, cached features, and models are stored under `data/` and `models/` (gitignored).
- For best results, provide thousands of locations spread across the target region.
- GPU (CUDA) is used automatically when available.