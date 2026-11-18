import io
import json
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
import rasterio
from openpyxl import Workbook
from pyproj import CRS
from pykrige import variogram_models
from pykrige.ok import OrdinaryKriging
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from diagnostic_report import findings, technical_pdf
from scipy.linalg import pinv
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist, pdist

COLORS = ["#e74c43", "#f29435", "#f4d64f", "#91ce53", "#22864b"]
LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
NODATA = -9999


def empirical(x, y, values, distance, count):
    d = pdist(np.column_stack((x, y)))
    semivariance = pdist(values[:, None], metric="sqeuclidean") / 2
    bins = np.minimum((d / distance).astype(int), count)
    indices = [bins == i for i in range(count) if np.any(bins == i)]
    return np.array([d[i].mean() for i in indices]), np.array([semivariance[i].mean() for i in indices])


def fit(x, y, values, settings, distance):
    if np.all(values == values[0]):
        return [0, 0] if settings["variogram_model"] == "linear" else [0, 1, 0]
    if not distance:
        return None
    model = settings["variogram_model"]
    lag, gamma = empirical(x, y, values, distance, settings["nlags"])
    size = 2 if model == "linear" else 3
    if len(lag) < size:
        raise ValueError("Increase lag distance or number of lags to include more sample pairs")
    maximum = max(gamma.max(), 1e-12)
    extent = max(lag.max(), 1)
    if model == "linear":
        initial, upper = [maximum / extent, maximum / 10], [np.inf, maximum]
    elif model == "power":
        initial, upper = [maximum / extent, 1, maximum / 10], [np.inf, 1.999, maximum]
    else:
        initial, upper = [maximum, extent / 2, maximum / 10], [np.inf, extent * 2, maximum]
    function = getattr(variogram_models, model + "_variogram_model")
    params = least_squares(lambda p: function(p, lag) - gamma, initial, bounds=(0, upper)).x.tolist()
    # PyKrige expects a full sill when variogram parameters are supplied as a list.
    if model not in {"linear", "power"}:
        params[0] += params[2]
    return params


def diagnostics(x, y, values, settings, distance, enabled, manual=""):
    model = settings["variogram_model"]
    if manual:
        params = [float(value) for value in manual.split(",")]
        if len(params) != (2 if model == "linear" else 3) or not all(np.isfinite(params)):
            raise ValueError("Enter all manual variogram parameters")
        if params[0] < 0 or params[-1] < 0 or (model != "linear" and params[1] <= 0):
            raise ValueError("Variogram values must be nonnegative and range or exponent must be positive")
        if model == "power" and params[1] >= 2:
            raise ValueError("Power exponent must be less than 2")
        if model not in {"linear", "power"} and params[0] < params[-1]:
            raise ValueError("Sill must be greater than or equal to nugget")
        if distance and not len(empirical(x, y, values, distance, settings["nlags"])[0]):
            raise ValueError("Increase lag distance or number of lags to include sample pairs")
    else:
        params = fit(x, y, values, settings, distance)
    if not enabled:
        return params, None
    predicted = np.empty_like(values, dtype=float)
    folds = np.random.default_rng(42).permutation(len(values))
    for test in np.array_split(folds, 5):
        train = np.ones(len(values), dtype=bool)
        train[test] = False
        args = x[train], y[train], values[train]
        if np.all(args[2] == args[2][0]):
            predicted[test] = args[2][0]
        else:
            fold = OrdinaryKriging(*args, **settings, variogram_parameters=params if manual else fit(*args, settings, distance))
            predicted[test] = fold.execute("points", x[test], y[test])[0]
    error = predicted - values
    total = ((values - values.mean()) ** 2).sum()
    return params, dict(folds=5, observed=values.tolist(), predicted=predicted.tolist(),
                        r2=float(1 - (error**2).sum() / total) if total else None,
                        rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error))),
                        mbe=float(error.mean()))


def predict_grid(kriging, prediction, variance, rows, cols, west, north, resolution):
    points = np.column_stack((kriging.X_ADJUSTED, kriging.Y_ADJUSTED))
    count = len(points)
    # Ordinary kriging system with one constraint row for weights summing to one.
    system = np.ones((count + 1, count + 1))
    variogram = kriging.variogram_function
    parameters = kriging.variogram_model_parameters
    system[:count, :count] = -variogram(parameters, cdist(points, points))
    np.fill_diagonal(system, 0)
    inverse = pinv(system)
    for start in range(0, len(rows), 2048):
        rr, cc = rows[start:start + 2048], cols[start:start + 2048]
        coordinates = np.column_stack((west + (cc + .5) * resolution, north - (rr + .5) * resolution))
        distances = cdist(coordinates, points)
        rhs = np.ones((len(rr), count + 1))
        rhs[:, :count] = np.where(distances <= kriging.eps, 0, -variogram(parameters, distances))
        weights = rhs @ inverse.T
        prediction[rr, cc] = weights[:, :count] @ kriging.Z
        variance[rr, cc] = np.maximum(-np.einsum("ij,ij->i", weights, rhs), 0)


def geotiff(array, transform, crs, nodata):
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                      dtype=array.dtype, crs=crs, transform=transform, nodata=nodata,
                      compress="deflate") as dataset:
            dataset.write(array, 1)
        return mem.read()


def preview(array, transform, crs):
    height, width = array.shape
    bounds = array_bounds(height, width, transform)
    _, w, h = calculate_default_transform(crs, "EPSG:3857", width, height, *bounds)
    destination = from_bounds(*transform_bounds(crs, "EPSG:3857", *bounds), w, h)
    missing = array == NODATA
    # Fill edge support for bilinear resampling. The browser clips to the exact
    # study polygon; these extra values are never used in GIS exports or stats.
    nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
    filled = array[tuple(nearest)]
    image = np.full((h, w), NODATA, dtype=array.dtype)
    reproject(filled, image, src_transform=transform, src_crs=crs, src_nodata=NODATA,
              dst_transform=destination, dst_crs="EPSG:3857", dst_nodata=NODATA,
              resampling=Resampling.bilinear)
    # A projected bounding rectangle can extend beyond the source at its corners.
    missing = image == NODATA
    if missing.any() and not missing.all():
        nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
        image[missing] = image[tuple(nearest[:, missing])]
    return geotiff(image, destination, "EPSG:3857", NODATA)


def reports(out, summary):
    columns = ["Class", "Range", "Area (ha)", "Share (%)"]
    rows = [[row["label"], row["range"], round(row["area_ha"], 3), round(row["percent"], 2)]
            for row in summary["classes"]]
    rows.append(["Total", "", round(sum(row["area_ha"] for row in summary["classes"]), 3), 100])
    info = [["Study area", summary["study_area"]], ["Parameter", summary["parameter"]],
            ["Units", summary["units"] or "Source units"], ["Method", "Ordinary kriging"],
            ["Variogram model", summary["model"]],
            ["Variogram fitting", summary["variogram_fit"]],
            ["Resolution (m)", summary["resolution_m"]], ["Samples", summary["samples"]],
            ["Classification", summary["classification"]]]
    metrics = [[key.upper(), round(summary["validation"][key], 6) if summary["validation"][key] is not None else "Undefined"]
               for key in ("r2", "rmse", "mae", "mbe")]
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Area statistics"
    for row in [*info, [], columns, *rows]:
        sheet.append(row)
        for cell in sheet[sheet.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"
    sheet.column_dimensions["A"].width = 25
    sheet.column_dimensions["B"].width = 32
    validation = workbook.create_sheet("5-fold validation")
    for row in [*metrics, [], ["Observed", "Predicted"], *zip(summary["validation"]["observed"], summary["validation"]["predicted"])]:
        validation.append(list(row))
    xlsx = io.BytesIO()
    workbook.save(xlsx)
    out.writestr("soil_report.xlsx", xlsx.getvalue())

    metric_names = ["Coefficient of determination (R²)", "Root mean squared error (RMSE)",
                    "Mean absolute error (MAE)", "Mean bias error (MBE)"]
    out.writestr("soil_report.pdf", technical_pdf(summary, info, columns, rows,
                 [[name, metric[1]] for name, metric in zip(metric_names, metrics)]))


def qgis_project(transform, shape, epsg):
    root = ET.Element("qgis", {"version": "3.40.0", "projectname": "Soil Fertility GIS"})
    ET.SubElement(root, "title").text = "Soil Fertility GIS"
    properties = ET.SubElement(root, "properties")
    spatial = ET.SubElement(properties, "SpatialRefSys")
    ET.SubElement(spatial, "ProjectionsEnabled", {"type": "int"}).text = "1"
    projection = CRS.from_user_input(epsg)
    crs = ET.SubElement(ET.SubElement(root, "projectCrs"), "spatialrefsys", {"nativeFormat": "Wkt"})
    for key, value in [("wkt", projection.to_wkt()), ("srid", str(projection.to_epsg())),
                       ("authid", epsg), ("description", projection.name), ("geographicflag", "false")]:
        ET.SubElement(crs, key).text = value
    bounds = dict(zip(("xmin", "ymin", "xmax", "ymax"), map(str, array_bounds(*shape, transform))))
    view = ET.SubElement(root, "ProjectViewSettings", {"rotation": "0", "UseProjectScales": "0"})
    extent = ET.SubElement(view, "DefaultViewExtent", bounds)
    extent.append(ET.fromstring(ET.tostring(crs)))
    tree = ET.SubElement(root, "layer-tree-group", {"name": "", "checked": "Qt::Checked", "expanded": "1"})
    layers = ET.SubElement(root, "projectlayers")
    for identifier, name, source, provider, checked in [
        ("classes", "Fertility classes", "./classification.tif", "gdal", True),
        ("prediction", "Prediction", "./prediction.tif", "gdal", False),
        ("variance", "Kriging variance", "./variance.tif", "gdal", False),
        ("samples", "Sample points", "./soil_layers.gpkg|layername=samples", "ogr", False),
        ("area", "Study area", "./soil_layers.gpkg|layername=study_area", "ogr", False),
    ]:
        ET.SubElement(tree, "layer-tree-layer", {"id": identifier, "name": name, "source": source,
                      "providerKey": provider, "checked": "Qt::Checked" if checked else "Qt::Unchecked", "expanded": "1"})
        layer = ET.SubElement(layers, "maplayer", {"type": "raster" if provider == "gdal" else "vector"})
        for key, val in [("id", identifier), ("datasource", source), ("layername", name), ("provider", provider)]:
            ET.SubElement(layer, key).text = val
        if identifier == "classes":
            renderer = ET.SubElement(ET.SubElement(layer, "pipe"), "rasterrenderer", {"type": "paletted", "band": "1", "opacity": "1", "alphaBand": "-1"})
            palette = ET.SubElement(renderer, "colorPalette")
            for i, (label, color) in enumerate(zip(LABELS, COLORS), 1):
                ET.SubElement(palette, "paletteEntry", {"value": str(i), "color": color, "label": label, "alpha": "255"})
    qgz = io.BytesIO()
    with zipfile.ZipFile(qgz, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("soil_fertility.qgs", b"<!DOCTYPE qgis>\n" + ET.tostring(root, encoding="utf-8"))
    return qgz.getvalue()


def write_outputs(folder, pts, area, z, v, mask, transform, summary, dashboard, thresholds, ok, dem, distance, lags):
    output = io.BytesIO()
    crs = pts.crs
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr("prediction.tif", geotiff(z, transform, crs, NODATA))
        out.writestr("variance.tif", geotiff(v, transform, crs, NODATA))
        out.writestr("municipalities.geojson", area.to_crs(4326).to_json())
        out.writestr("samples.geojson", pts.to_crs(4326).to_json())
        if dashboard:
            vals = z[mask]
            breaks = np.array([float(value) for value in thresholds.split(",")]) if thresholds.strip() else np.linspace(vals.min(), vals.max(), 6)[1:-1]
            if len(breaks) != 4 or not np.isfinite(breaks).all() or (thresholds.strip() and not (np.diff(breaks) > 0).all()):
                raise ValueError("Enter four increasing numeric class thresholds")
            stats = []
            classified = np.zeros(mask.shape, dtype="uint8")
            classified[mask] = np.searchsorted(breaks, vals, side="right") + 1
            def limit(value):
                return f"{value:.6g}"
            ranges = [f"< {limit(breaks[0])}", *[f"{limit(a)} to < {limit(b)}" for a, b in zip(breaks, breaks[1:])], f"≥ {limit(breaks[-1])}"]
            for i, (label, color, range_label) in enumerate(zip(LABELS, COLORS, ranges), 1):
                count = int((classified == i).sum())
                stats.append(dict(label=label, color=color, range=range_label, area_ha=count * summary["resolution_m"]**2 / 10000, percent=count / len(vals) * 100))
            lag, gamma = empirical(ok.X_ORIG, ok.Y_ORIG, ok.Z, distance, lags) if distance else (ok.lags, ok.semivariance)
            line = np.linspace(0, max(lag, default=distance * lags), 80)
            summary.update(classes=stats, classification="Custom thresholds" if thresholds.strip() else "Equal intervals from predicted values",
                           breaks=breaks.tolist(), variance=[float(v[mask].min()), float(v[mask].max())],
                           variogram=dict(lag=lag.tolist(), gamma=gamma.tolist(), distance=line.tolist(),
                                          fitted=ok.variogram_function(ok.variogram_model_parameters, line).tolist()))
            out.writestr("classification.tif", geotiff(classified, transform, crs, 0))
            out.writestr("map_prediction.tif", preview(z, transform, crs))
            outline = area.to_crs(4326).geometry.make_valid().union_all()
            out.writestr("map_boundary.geojson", json.dumps(outline.__geo_interface__))
            out.writestr("map_variance.tif", preview(v, transform, crs))
            if dem:
                if not (dem.filename or "").lower().endswith((".tif", ".tiff")):
                    raise ValueError("Upload the DEM as a GeoTIFF")
                with MemoryFile(dem.file.read()) as mem, mem.open() as source:
                    if not source.crs:
                        raise ValueError("DEM must have a coordinate reference system")
                    elevation = np.full(z.shape, NODATA, dtype="float32")
                    reproject(rasterio.band(source, 1), elevation, dst_crs=crs, dst_transform=transform, dst_nodata=NODATA)
                elevation[~mask] = NODATA
                valid = np.isfinite(elevation) & (elevation != NODATA)
                elevation[~valid] = NODATA
                if not valid.any():
                    raise ValueError("DEM does not overlap the study area")
                summary["elevation"] = [float(elevation[valid].min()), float(elevation[valid].max())]
                out.writestr("map_elevation.tif", preview(elevation, transform, crs))
            gpkg = folder / "soil_layers.gpkg"
            shp = folder / "shapefiles"
            shp.mkdir()
            for name, layer in [("samples", pts), ("study_area", area)]:
                layer.to_file(gpkg, layer=name, driver="GPKG", index=False)
                layer.to_file(shp / (name + ".shp"), index=False)
            out.write(gpkg, gpkg.name)
            shape_zip = io.BytesIO()
            with zipfile.ZipFile(shape_zip, "w", zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(shp.iterdir()):
                    archive.write(file, file.name)
            out.writestr("soil_layers_shp.zip", shape_zip.getvalue())
            out.writestr("soil_fertility.qgz", qgis_project(transform, z.shape, crs.to_string()))
            summary["findings"] = findings(summary)
            reports(out, summary)
        else:
            summary["warning"] = "No cross-validation performed"
        out.writestr("summary.json", json.dumps(summary, allow_nan=False))
    output.seek(0)
    return output
