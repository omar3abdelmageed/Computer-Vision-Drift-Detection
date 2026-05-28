from __future__ import annotations

from pathlib import Path

from backend.common import ImageFeatures, ImageMetadata, SplitName
from backend.utils.hashing import file_sha256


def load_image_rgb(path: Path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def extract_image_metadata(path: Path, split: SplitName | None = None) -> ImageMetadata:
    from PIL import Image
    with Image.open(path) as img:
        width, height = img.size
        mode = img.mode
        image_format = img.format or path.suffix.lstrip(".").upper()
        channels = len(img.getbands())
    return ImageMetadata(
        path=path,
        filename=path.name,
        split=split,
        content_hash=file_sha256(path),
        image_format=image_format,
        color_mode=mode,
        num_channels=channels,
        bit_depth=None,
        width=width,
        height=height,
        aspect_ratio=width / height if height else 0.0,
        file_size_bytes=path.stat().st_size,
    )


def extract_image_features(path: Path) -> ImageFeatures:
    import numpy as np
    from PIL import Image, ImageFilter, ImageStat

    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype("float32") / 255.0
    gray = image.convert("L")
    gray_arr = np.asarray(gray).astype("float32") / 255.0
    stat = ImageStat.Stat(image)
    rgb_means = [float(v / 255.0) for v in stat.mean]
    rgb_stds = [float(v / 255.0) for v in stat.stddev]
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_density = float((np.asarray(edges) > 25).mean())
    saturation = arr.max(axis=2) - arr.min(axis=2)
    embedding = [
        float(gray_arr.mean()),
        float(gray_arr.std()),
        float(gray_arr.max() - gray_arr.min()),
        float(saturation.mean()),
        float(saturation.std()),
        edge_density,
        *rgb_means,
        *rgb_stds,
    ]
    return ImageFeatures(
        brightness_mean=float(gray_arr.mean()),
        brightness_std=float(gray_arr.std()),
        contrast=float(gray_arr.std()),
        sharpness=float(np.asarray(gray.filter(ImageFilter.FIND_EDGES)).var()),
        saturation_mean=float(saturation.mean()),
        saturation_std=float(saturation.std()),
        edge_density=edge_density,
        rgb_channel_means=rgb_means,
        rgb_channel_stds=rgb_stds,
        embedding=embedding,
    )
