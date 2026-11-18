"""Print layout for a single fertility prediction map."""

import io
import json
import math
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import shapely
from PIL import Image, ImageColor, ImageDraw, ImageFont
from pyproj import Geod, Transformer
from rasterio import band
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from shapely.geometry import shape

from classification import classify_values
from coast import CREDIT, OCEAN, PROJECT, SHORELINE, draw_shoreline, paint_ocean

SCALE = 3840 / 2360


def px(value):
    return round(value * SCALE)


MAP_SIZE = (px(1552), px(1340))
PAGE_SIZE = (3840, px(1500))
GREEN = "#005b3c"
INK = "#202d27"
RULE = "#b6c0b9"
MERCATOR_EDGE = 20037508.342789244
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile"
TERRAIN_CREDIT = "World Hillshade © Esri and contributors (esri.com)"
TERRAIN_MAX_TILE_BYTES = 512 * 1024
LAYOUT_MAX_BYTES = 12 * 1024 * 1024


@lru_cache(maxsize=128)
def hillshade_tile(z, x, y):
    with urlopen(f"{TILE_URL}/{z}/{y}/{x}?blankTile=false", timeout=4) as response:
        data = response.read(TERRAIN_MAX_TILE_BYTES + 1)
    if len(data) > TERRAIN_MAX_TILE_BYTES:
        raise ValueError("Hillshade tile exceeded size limit")
    with Image.open(io.BytesIO(data)) as image:
        if image.size != (256, 256):
            raise ValueError("Unexpected hillshade tile dimensions")
        return image.convert("RGB")


def hillshade(bounds):
    west, south, east, north = bounds
    span = 2 * MERCATOR_EDGE
    zoom = max(0, min(16, math.floor(math.log2(span / max(east - west, north - south) * 7))))
    best = None
    best_coverage = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        for z in range(zoom, max(-1, zoom - 5), -1):
            n = 2 ** z
            factor = n * 256 / span
            left, right = (west + MERCATOR_EDGE) * factor, (east + MERCATOR_EDGE) * factor
            top, bottom = (MERCATOR_EDGE - north) * factor, (MERCATOR_EDGE - south) * factor
            x0, y0 = math.floor(left / 256), math.floor(top / 256)
            x1, y1 = math.ceil(right / 256), math.ceil(bottom / 256)
            tiles = [(x, y) for x in range(x0, x1) for y in range(max(0, y0), min(n, y1))]

            def load(tile):
                x, y = tile
                try:
                    return hillshade_tile(z, x % n, y)
                except (OSError, URLError, ValueError):
                    return None

            images = list(pool.map(load, tiles))
            coverage = sum(image is not None for image in images) / len(tiles) if tiles else 0
            if coverage <= best_coverage:
                continue
            mosaic = Image.new("RGB", ((x1 - x0) * 256, (y1 - y0) * 256), "#eeefec")
            for (x, y), image in zip(tiles, images):
                if image is not None:
                    mosaic.paste(image, ((x - x0) * 256, (y - y0) * 256))
            crop = (left - x0 * 256, top - y0 * 256, right - x0 * 256, bottom - y0 * 256)
            best = mosaic.resize(MAP_SIZE, Image.Resampling.BILINEAR, crop)
            best_coverage = coverage
            if coverage == 1:
                return best, TERRAIN_CREDIT
    if best is not None:
        return best, TERRAIN_CREDIT + " | Partial coverage"
    return Image.new("RGB", MAP_SIZE, "#eeefec"), "Hillshade unavailable"


@lru_cache(maxsize=40)
def font(size, bold=False):
    size = px(size)
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


class LayoutDraw:
    """Scale page coordinates while drawing directly into the 4K image."""

    def __init__(self, image):
        self.draw = ImageDraw.Draw(image)

    @staticmethod
    def coordinates(values):
        if isinstance(values[0], (tuple, list)):
            return [tuple(px(value) for value in point) for point in values]
        return tuple(px(value) for value in values)

    def rectangle(self, xy, *, width=1, **kwargs):
        self.draw.rectangle(self.coordinates(xy), width=px(width), **kwargs)

    def ellipse(self, xy, *, width=1, **kwargs):
        self.draw.ellipse(self.coordinates(xy), width=px(width), **kwargs)

    def line(self, xy, *, width=1, **kwargs):
        self.draw.line(self.coordinates(xy), width=px(width), **kwargs)

    def polygon(self, xy, **kwargs):
        self.draw.polygon(self.coordinates(xy), **kwargs)

    def text(self, xy, text, **kwargs):
        self.draw.text(self.coordinates(xy), text, **kwargs)

    def textlength(self, text, **kwargs):
        return self.draw.textlength(text, **kwargs) / SCALE


def fit_text(draw, position, text, width, size=26, bold=False, fill=INK, anchor="lm"):
    while size > 12 and draw.textlength(text, font=font(size, bold)) > width:
        size -= 1
    while len(text) > 1 and draw.textlength(text, font=font(size, bold)) > width:
        text = text[:-2] + "…"
    draw.text(position, text, font=font(size, bold), fill=fill, anchor=anchor)


def view_bounds(geometry):
    west, south, east, north = geometry.bounds
    width, height = east - west, north - south
    width = max(width, height * MAP_SIZE[0] / MAP_SIZE[1]) * 1.14
    height = width * MAP_SIZE[1] / MAP_SIZE[0]
    x, y = (west + east) / 2, (south + north) / 2
    return x - width / 2, y - height / 2, x + width / 2, y + height / 2


def draw_scale(draw, origin, bounds):
    west, south, east, north = bounds
    center_x, center_y = (west + east) / 2, (south + north) / 2
    geographic = Transformer.from_crs(3857, 4326, always_xy=True)
    lon, lat = geographic.transform(center_x, center_y)
    stop, _ = geographic.transform(center_x + (east - west) * px(230) / MAP_SIZE[0], center_y)
    maximum = Geod(ellps="WGS84").inv(lon, lat, stop, lat)[2]
    base = 10 ** math.floor(math.log10(maximum))
    metres = max(base * step for step in (1, 2, 5) if base * step <= maximum)
    pixels = round(230 * metres / maximum)
    x, y = origin
    draw.rectangle((x - 17, y - 40, x + pixels + 78, y + 36), fill="white", outline=RULE)
    draw.rectangle((x, y, x + pixels, y + 10), fill="white", outline=INK, width=2)
    draw.rectangle((x, y, x + pixels // 2, y + 10), fill=INK)
    label = f"{metres / 1000:g} km" if metres >= 1000 else f"{metres:g} m"
    draw.text((x, y - 18), "0", font=font(17), fill=INK, anchor="mm")
    draw.text((x + pixels, y - 18), label, font=font(17), fill=INK, anchor="mm")
    draw.text((x, y + 23), "Scale at map center", font=font(13), fill=INK, anchor="lm")


def read_member(archive, name, maximum):
    if archive.getinfo(name).file_size > maximum:
        raise ValueError(f"{name} exceeds layout limits")
    return archive.read(name)


def map_layout(package):
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        summary = json.loads(read_member(archive, "summary.json", 192_000))
        outline = shape(json.loads(read_member(archive, "map_boundary.geojson", 6_000_000)))
        samples = json.loads(read_member(archive, "samples.geojson", 4_000_000))
        raster = read_member(archive, "map_prediction.tif", 3_000_000)
    classes = summary["classes"]
    breaks = np.asarray(summary["breaks"], dtype=float)
    if len(classes) not in {4, 5} or breaks.shape not in {(4,), (5,)} or not np.isfinite(breaks).all():
        raise ValueError("The result must have four or five fertility classes")
    if outline.is_empty or not shapely.get_coordinates(outline).shape[0] or any(not math.isfinite(v) for v in outline.bounds):
        raise ValueError("Invalid study boundary")
    if not shapely.box(100, -5, 140, 30).covers(outline):
        raise ValueError("The layout coastline dataset covers the Philippines and surrounding waters")
    geometry = shapely.transform(outline, PROJECT.transform, interleaved=False)
    bounds = view_bounds(geometry)
    if not all(math.isfinite(v) for v in bounds) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Invalid layout extent")

    width, height = MAP_SIZE
    transform = from_bounds(*bounds, width, height)
    data = np.full((height, width), -9999, dtype="float32")
    with MemoryFile(raster) as memory, memory.open() as source:
        if source.crs is None or source.width * source.height > 520_000:
            raise ValueError("Invalid prediction preview")
        reproject(band(source, 1), data, dst_transform=transform, dst_crs=3857,
                  dst_nodata=-9999, resampling=Resampling.bilinear)
    mask = geometry_mask([geometry], out_shape=data.shape, transform=transform, invert=True)
    valid = mask & np.isfinite(data) & (data != -9999)
    if not valid.any():
        raise ValueError("No predicted cells intersect the layout")
    palette = np.array([(0, 0, 0), *[ImageColor.getrgb(row["color"]) for row in classes]], dtype="uint8")
    rgba = np.empty((*data.shape, 4), dtype="uint8")
    for row in range(0, height, 128):
        chunk = slice(row, row + 128)
        category = classify_values(data[chunk], summary)
        rgba[chunk, :, :3] = palette[category]
        rgba[chunk, :, 3] = np.where(valid[chunk] & (category != 0), 179, 0)
    del data, mask, valid
    background, relief = hillshade(bounds)
    shoreline = paint_ocean(background, bounds)
    atlas = background.convert("RGBA")
    atlas.alpha_composite(Image.fromarray(rgba))
    del rgba, background
    draw_shoreline(atlas, bounds, shoreline, width=px(2))
    draw_boundaries(atlas, geometry, samples, transform)
    return compose(atlas, bounds, summary, relief)


def draw_boundaries(atlas, geometry, samples, transform):
    draw = ImageDraw.Draw(atlas)
    pixels = ~transform
    for part in shapely.get_parts(geometry):
        if part.geom_type != "Polygon":
            continue
        for ring in [part.exterior, *part.interiors]:
            coords = [pixels * (x, y) for x, y in ring.coords]
            draw.line(coords, fill="white", width=px(3))
            draw.line(coords, fill=INK, width=px(1))
    for feature in samples["features"][:500]:
        lon, lat = feature["geometry"]["coordinates"][:2]
        x, y = pixels * PROJECT.transform(lon, lat)
        radius = px(5)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#fff9e9", outline=INK, width=px(2))


def compose(atlas, bounds, summary, relief):
    page = Image.new("RGB", PAGE_SIZE, "white")
    draw = LayoutDraw(page)
    draw.rectangle((20, 20, 2340, 1480), outline=INK, width=2)
    page.paste(atlas, (px(44), px(45)))
    draw.rectangle((43, 44, 1597, 1386), outline=RULE, width=2)
    draw.rectangle((1464, 65, 1575, 203), fill="white", outline=RULE)
    draw.text((1519, 90), "N", font=font(25, True), fill=INK, anchor="mm")
    draw.polygon([(1519, 111), (1484, 182), (1519, 163), (1554, 182)], fill=INK)
    draw.polygon([(1518, 123), (1492, 175), (1518, 162)], fill="white")
    draw_scale(draw, (79, 1330), bounds)
    fit_text(draw, (45, 1415), relief + " | " + CREDIT, 1530, 19)
    fit_text(draw, (45, 1450), "Study boundary and sample locations shown. Classified areas are estimated from raster cells.", 1530, 19)

    x0, x1 = 1640, 2318

    def band(y, title, height=54):
        draw.rectangle((x0, y, x1, y + height), fill=GREEN)
        fit_text(draw, ((x0 + x1) / 2, y + height / 2), title, x1-x0-32, 25, True, "white", "mm")

    band(45, "SOIL FERTILITY MAP", 76)
    fit_text(draw, ((x0 + x1)/2, 159), summary.get("study_area") or summary["province"], x1 - x0 - 20, 34, True, anchor="mm")
    names = {"N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)", "pH": "Soil pH", "OM": "Organic matter (OM)"}
    parameter = names.get(summary["parameter"], summary["parameter"])
    fit_text(draw, ((x0 + x1)/2, 204), f"{parameter} | {summary['units'] or 'Source units'}", x1-x0-20, 23, anchor="mm")
    band(242, "FERTILITY CLASS LEGEND")
    for i, row in enumerate(summary["classes"]):
        y = 322 + i * 46
        draw.rectangle((x0 + 22, y - 13, x0 + 64, y + 13), fill=row["color"], outline=RULE)
        fit_text(draw, (x0 + 82, y), row["label"], 186, 23)
        fit_text(draw, (x0 + 278, y), row["range"], x1-x0-292, 23)
    draw.line((x0+17, 527, x1-17, 527), fill=RULE, width=2)
    draw.rectangle((x0+22, 550, x0+64, 575), fill=OCEAN, outline=SHORELINE)
    draw.text((x0+82, 562), "Ocean", font=font(21), fill=INK, anchor="lm")
    draw.line((x0+22, 609, x0+64, 609), fill=SHORELINE, width=3)
    draw.text((x0+82, 609), "Shoreline", font=font(21), fill=INK, anchor="lm")
    draw.line((x0+352, 562, x0+394, 562), fill=INK, width=2)
    draw.text((x0+412, 562), "Study boundary", font=font(21), fill=INK, anchor="lm")
    draw.ellipse((x0+366, 602, x0+380, 616), fill="#fff9e9", outline=INK, width=2)
    draw.text((x0+412, 609), "Sample point", font=font(21), fill=INK, anchor="lm")
    band(651, "CLASSIFIED AREA SUMMARY")

    classes = summary["classes"]
    total = sum(float(row["area_ha"]) for row in classes)
    values = [["Class", "Area (ha)", "Share (%)"],
              *[[row["label"], f"{float(row['area_ha']):,.2f}", f"{float(row['percent']):.2f}"] for row in classes],
              ["Total", f"{total:,.2f}", f"{100 if total else 0:.2f}"]]
    edges = [x0, x0 + 220, x0 + 440, x1]
    for i, row in enumerate(values):
        y = 720 + 46 * i
        draw.rectangle((x0, y, x1, y+46), fill="#e8eee9" if i in (0, len(values) - 1) else "white", outline=RULE)
        for j, value in enumerate(row):
            fit_text(draw, ((edges[j]+edges[j+1])/2, y+23), value, edges[j+1]-edges[j]-12, 22, i in (0, len(values) - 1), anchor="mm")
    band(1072, "MAP INFORMATION")
    for i, line in enumerate([
        f"Method: Ordinary kriging · {summary['model'].title()} variogram",
        f"Grid: {summary['resolution_m']} m · {summary['samples']} samples",
        f"Analysis CRS: {summary['crs']} | Map CRS: EPSG:3857",
        f"Classification: {summary['classification']}",
        summary.get("classification_source", "Generated by Soil Fertility Map Generator"),
    ]):
        fit_text(draw, (x0+18, 1156+i*48), line, x1-x0-36, 22)
    if summary.get("classification_note"):
        fit_text(draw, (x0+18, 1415), f"Classified coverage: {summary.get('classified_coverage_pct', 100):.1f}% of prediction grid.", x1-x0-36, 20)
    output = io.BytesIO()
    page.save(output, format="JPEG", quality=96, subsampling=0, dpi=(300, 300))
    return output.getvalue()
