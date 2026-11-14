import math
import io
import os
import requests
from PIL import Image, ImageDraw

OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
HEADERS = {"User-Agent": "GeoGuesserTrainer/1.0 (educational)"}


def latlng_to_tile(lat, lng, z):
    n = 2 ** z
    x = (lng + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(z, x, y):
    url = OSM_TILE_URL.format(z=z, x=x, y=y)
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def static_map(lat, lng, z=15, size=512):
    cx_float, cy_float = latlng_to_tile(lat, lng, z)
    cx = int(math.floor(cx_float))
    cy = int(math.floor(cy_float))
    grid = 3
    canvas = Image.new("RGB", (grid * 256, grid * 256))
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            try:
                tile = fetch_tile(z, cx + dx, cy + dy)
            except Exception:
                tile = Image.new("RGB", (256, 256), (200, 200, 200))
            canvas.paste(tile, ((dx + 1) * 256, (dy + 1) * 256))
    px = (cx_float - (cx - 1)) * 256
    py = (cy_float - (cy - 1)) * 256
    draw = ImageDraw.Draw(canvas)
    r = 8
    draw.ellipse((px - r, py - r, px + r, py + r), fill=(220, 20, 60), outline=(255, 255, 255), width=2)
    half = size // 2
    left = int(px - half)
    top = int(py - half)
    left = max(0, min(canvas.width - size, left))
    top = max(0, min(canvas.height - size, top))
    cropped = canvas.crop((left, top, left + size, top + size))
    return cropped


def leaflet_html(lat, lng, zoom=None, radius_km=None):
    if zoom is None:
        if not radius_km:
            zoom = 15
        elif radius_km <= 2:
            zoom = 14
        elif radius_km <= 10:
            zoom = 12
        elif radius_km <= 40:
            zoom = 10
        elif radius_km <= 150:
            zoom = 8
        elif radius_km <= 600:
            zoom = 6
        elif radius_km <= 2500:
            zoom = 4
        else:
            zoom = 3
    circle_js = ""
    if radius_km and radius_km > 0:
        circle_js = (
            f"L.circle([{lat}, {lng}], {{radius: {radius_km * 1000:.0f}, "
            "color: '#d6336c', weight: 2, opacity: 0.8, "
            "fillColor: '#d6336c', fillOpacity: 0.08}).addTo(map);"
        )
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{{margin:0}}#map{{height:100vh;width:100vw}}</style>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map').setView([{lat}, {lng}], {zoom});
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19, attribution: '&copy; OpenStreetMap'
}}).addTo(map);
{circle_js}
L.marker([{lat}, {lng}]).addTo(map).bindPopup('Predicted location').openPopup();
</script>
</body></html>"""


def training_map_html(locations):
    pts = []
    for loc in locations:
        try:
            pts.append((float(loc["lat"]), float(loc["lng"])))
        except Exception:
            continue
    markers_js = ",".join(f"[{la},{lo}]" for la, lo in pts)
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<style>body{{margin:0}}#map{{height:600px;width:100%}}</style>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map').setView([39.5, -98.35], 4);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19, attribution: '&copy; OpenStreetMap'
}}).addTo(map);
var markers = L.markerClusterGroup();
var pts = [{markers_js}];
for (var i=0;i<pts.length;i++) {{
  var m = L.marker(pts[i]);
  var ll = pts[i];
  (function(marker, la, lo) {{
    marker.on('click', function() {{
      window.open('https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=' + la + ',' + lo, '_blank');
    }});
  }})(m, ll[0], ll[1]);
  markers.addLayer(m);
}}
map.addLayer(markers);
</script>
</body></html>"""
