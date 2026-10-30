import asyncio
from .downloader import prepare_dataset as _prepare


def prepare_dataset_sync(locations, views=4, zoom=2, view_size=224, keep_panos=False, jpeg_quality=85, progress_cb=None):
    return asyncio.run(_prepare(locations, views=views, zoom=zoom, view_size=view_size, keep_panos=keep_panos, jpeg_quality=jpeg_quality, progress_cb=progress_cb))
