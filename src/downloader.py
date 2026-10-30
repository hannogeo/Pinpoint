import asyncio
import os
import math
import numpy as np
import aiohttp
from PIL import Image

import streetlevel.streetview as sl_sv

from .config import IMAGES_DIR


def equirect_to_perspective(equi, yaw_deg=0.0, pitch_deg=0.0, fov_deg=90.0, out_w=256, out_h=256):
    """Render a perspective view from an equirectangular panorama."""
    equi = equi.convert("RGB")
    pano = np.asarray(equi, dtype=np.float32)  # H, W, 3
    pH, pW = pano.shape[:2]

    # output pixel grid
    xs = np.linspace(-1.0, 1.0, out_w)
    ys = np.linspace(1.0, -1.0, out_h)
    # account for aspect: fov horizontal
    tan_half = math.tan(math.radians(fov_deg) / 2.0)
    grid_x, grid_y = np.meshgrid(xs, ys)  # H, W
    cam_x = grid_x * tan_half
    cam_y = grid_y * tan_half * (out_h / out_w)
    cam_z = np.ones_like(grid_x)
    # normalize
    norm = np.sqrt(cam_x**2 + cam_y**2 + cam_z**2)
    rx = cam_x / norm
    ry = cam_y / norm
    rz = cam_z / norm

    # pitch rotation around X
    pr = math.radians(pitch_deg)
    cp, sp = math.cos(pr), math.sin(pr)
    # yaw rotation around Y
    yr = math.radians(yaw_deg)
    cy, sy = math.cos(yr), math.sin(yr)

    # world = R_yaw @ R_pitch @ cam
    # first pitch: (rx, ry*cos - rz*sin? let's do standard)
    # camera coords: x right, y up, z forward
    # pitch around x: y'=y*cos - z*sin ; z'=y*sin + z*cos
    y1 = ry * cp - rz * sp
    z1 = ry * sp + rz * cp
    x1 = rx
    # yaw around y (up): x''=x*cos + z*sin ; z''=-x*sin + z*cos
    wx = x1 * cy + z1 * sy
    wy = y1
    wz = -x1 * sy + z1 * cy

    # to lat/lng
    lat = np.arcsin(np.clip(wy, -1.0, 1.0))
    lng = np.arctan2(wz, wx)

    # sample equirectangular
    u = (lng / (2 * math.pi) + 0.5) * (pW - 1)
    v = (0.5 - lat / math.pi) * (pH - 1)
    u = np.clip(u, 0, pW - 1)
    v = np.clip(v, 0, pH - 1)

    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = np.clip(u0 + 1, 0, pW - 1)
    v1 = np.clip(v0 + 1, 0, pH - 1)
    du = (u - u0)[..., None]
    dv = (v - v0)[..., None]

    I00 = pano[v0, u0]
    I10 = pano[v0, u1]
    I01 = pano[v1, u0]
    I11 = pano[v1, u1]
    out = (I00 * (1 - du) * (1 - dv) +
           I10 * du * (1 - dv) +
           I01 * (1 - du) * dv +
           I11 * du * dv)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


async def prepare_dataset(locations, views=4, zoom=2, view_size=224, keep_panos=False, jpeg_quality=85, progress_cb=None):
    """Download panoramas and generate perspective views. Returns list of samples (dicts with image,lat,lng)."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    samples = []
    total = len(locations)
    done = 0
    sem = asyncio.Semaphore(8)

    async with aiohttp.ClientSession() as session:
        async def process(loc):
            nonlocal done
            async with sem:
                pano_id = loc["panoId"]
                pano_path = os.path.join(IMAGES_DIR, f"{pano_id}.jpg")
                # If all views already exist, skip download entirely
                expected_views = [os.path.join(IMAGES_DIR, f"{pano_id}_v{i}.jpg") for i in range(views)]
                if all(os.path.exists(v) for v in expected_views):
                    lat = float(loc["lat"]); lng = float(loc["lng"])
                    out_samples = [{"image": v, "lat": lat, "lng": lng, "panoId": pano_id} for v in expected_views]
                    done += 1
                    if progress_cb:
                        progress_cb(done, total, f"cached {pano_id} ({len(out_samples)} views)")
                    return out_samples
                if not os.path.exists(pano_path):
                    try:
                        pano = await sl_sv.find_panorama_by_id_async(pano_id, session)
                        if pano is None:
                            done += 1
                            if progress_cb:
                                progress_cb(done, total, f"skip {pano_id}")
                            return []
                        await sl_sv.download_panorama_async(pano, pano_path, session, zoom=zoom)
                    except Exception as e:
                        done += 1
                        if progress_cb:
                            progress_cb(done, total, f"err {pano_id}: {e}")
                        return []
                # load pano
                try:
                    img = Image.open(pano_path)
                except Exception:
                    done += 1
                    if progress_cb:
                        progress_cb(done, total, f"badimg {pano_id}")
                    return []
                lat = float(loc["lat"])
                lng = float(loc["lng"])
                heading = float(loc.get("heading") or 0.0)
                # view yaws
                if views <= 1:
                    yaws = [heading]
                else:
                    yaws = [i * (360.0 / views) for i in range(views)]
                out_samples = []
                for i, yaw in enumerate(yaws):
                    view_path = os.path.join(IMAGES_DIR, f"{pano_id}_v{i}.jpg")
                    if not os.path.exists(view_path):
                        try:
                            vimg = equirect_to_perspective(img, yaw_deg=yaw, pitch_deg=0.0, fov_deg=90.0, out_w=view_size, out_h=view_size)
                            vimg.save(view_path, quality=jpeg_quality)
                        except Exception as e:
                            continue
                    out_samples.append({"image": view_path, "lat": lat, "lng": lng, "panoId": pano_id})
                try:
                    img.close()
                except Exception:
                    pass
                if not keep_panos and os.path.exists(pano_path):
                    try:
                        os.remove(pano_path)
                    except Exception:
                        pass
                done += 1
                if progress_cb:
                    progress_cb(done, total, f"ok {pano_id} ({len(out_samples)} views)")
                return out_samples

        tasks = [process(loc) for loc in locations]
        results = await asyncio.gather(*tasks)
        for r in results:
            samples.extend(r)
    return samples
