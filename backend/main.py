import io, json, tempfile, zipfile, os, functools
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pykrige.ok import OrdinaryKriging
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import mapping
import boto3
from botocore.config import Config
app=FastAPI(title="Soil Fertility GIS API")
app.add_middleware(CORSMiddleware,allow_origins=["https://warren3398.github.io","http://localhost:8000"],allow_methods=["*"],allow_headers=["*"])
@app.get("/", include_in_schema=False)
def website():
    return FileResponse(Path(__file__).resolve().parents[1] / "index.html")

@app.get("/health")
def health(): return {"status":"ok","method":"ordinary_kriging","boundaries":"r2_catalog"}
def loadshape(b,folder):
    Path(folder).mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        for n in z.namelist():
            p=Path(n)
            if p.is_absolute() or ".." in p.parts: raise ValueError("Unsafe ZIP path")
        z.extractall(folder)
    paths=list(Path(folder).rglob("*.shp"))
    if len(paths)!=1: raise ValueError("ZIP must contain exactly one .shp")
    g=gpd.read_file(paths[0])
    if g.crs is None: raise ValueError("Shapefile must include a valid .prj")
    return g

def r2client():
    required=["R2_ACCOUNT_ID","R2_ACCESS_KEY_ID","R2_SECRET_ACCESS_KEY"]
    missing=[k for k in required if not os.environ.get(k)]
    if missing: raise HTTPException(503,"R2 boundary storage not configured: "+", ".join(missing))
    return boto3.client("s3",endpoint_url="https://"+os.environ["R2_ACCOUNT_ID"]+".r2.cloudflarestorage.com",aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],region_name="auto",config=Config(signature_version="s3v4"))
def read_boundary(key, *, ignore_geometry=False):
    bucket = os.environ.get("R2_BUCKET", "slopemap-gis-data")
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "boundary.gpkg")
            r2client().download_file(bucket, key, path)
            return gpd.read_file(path, ignore_geometry=ignore_geometry)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, "Cannot load administrative boundaries from R2; check object key and access") from exc


def field(frame, *aliases):
    columns = {str(column).strip().casefold(): column for column in frame.columns}
    return next((columns[alias] for alias in aliases if alias in columns), None)


@functools.lru_cache(maxsize=1)
def admin_catalog():
    key = os.environ.get("R2_MUNICIPALITY_BOUNDARY_KEY", "boundaries/municipalities.gpkg")
    municipalities = read_boundary(key)
    if municipalities.crs is None:
        raise HTTPException(503, "Boundary layer has no CRS")

    municipality_field = field(
        municipalities, "municipality", "mun_name", "adm3_en", "adm3_name",
        "city_mun", "city_name", "city", "name_3", "name_2",
    )
    if municipality_field is None:
        raise HTTPException(503, "Municipality name field missing: " + ", ".join(municipalities.columns))

    province_aliases = ("province", "prov_name", "prov", "adm2_en", "adm2_name")
    province_field = field(municipalities, *province_aliases)
    if province_field is None:
        # NAME_2 is a province only when the municipality comes from NAME_3.
        aliases = ("name_2", "name_1") if municipality_field.casefold() == "name_3" else ("name_1",)
        province_field = field(municipalities, *aliases)

    if province_field is not None:
        province_names = municipalities[province_field]
    else:
        code_aliases = ("adm2_pcode", "prov_pcode", "province_code", "psgc_prov")
        child_code = field(municipalities, *code_aliases)
        if child_code is None:
            raise HTTPException(503, "Province name or parent code missing: " + ", ".join(municipalities.columns))
        province_key = os.environ.get("R2_PROVINCE_BOUNDARY_KEY", "boundaries/provinces.gpkg")
        provinces = read_boundary(province_key, ignore_geometry=True)
        parent_code = field(provinces, *code_aliases)
        parent_name = field(provinces, *province_aliases, "name_2", "name_1")
        if parent_code is None or parent_name is None:
            raise HTTPException(503, "Province catalog requires a province name and code")
        codes = provinces[parent_code].astype("string").str.strip().str.upper()
        lookup = dict(zip(codes, provinces[parent_name]))
        province_names = municipalities[child_code].astype("string").str.strip().str.upper().map(lookup)

    catalog = municipalities.assign(
        province=province_names.astype("string").str.strip(),
        municipality=municipalities[municipality_field].astype("string").str.strip(),
    )
    if catalog.province.fillna("").eq("").any() or catalog.municipality.fillna("").eq("").any():
        raise HTTPException(503, "Some boundaries have no province or municipality name")
    return catalog

@app.get("/provinces")
def provinces():
    g=admin_catalog()
    return {"provinces":sorted(g.province.dropna().astype(str).unique().tolist())}
@app.get("/municipalities")
def municipalities(province:str):
    g=admin_catalog()
    a=g[g.province.astype(str).str.casefold()==province.casefold()]
    if a.empty: raise HTTPException(404,"Province not found")
    return json.loads(a.to_crs(4326).to_json())

@app.post("/interpolate")
async def interpolate(points:UploadFile=File(...),results:UploadFile=File(...),province:str=Form(...),parameter:str=Form("N"),resolution:int=Form(100)):
    if resolution<10 or resolution>5000: raise HTTPException(400,"Resolution must be 10-5000 m")
    try:
        with tempfile.TemporaryDirectory() as d:
            pts=loadshape(await points.read(),Path(d)/"points")
            bounds=admin_catalog().copy()
            lab=pd.read_csv(io.BytesIO(await results.read()))
            for df,cols in [(pts,["sample_id"]),(lab,["sample_id",parameter]),(bounds,["province","municipality"])]:
                missing=[x for x in cols if x not in df.columns]
                if missing: raise ValueError("Missing fields: "+", ".join(missing))
            if pts.geometry.geom_type.ne("Point").any(): raise ValueError("Sampling layer must contain points")
            area=bounds[bounds.province.astype(str).str.casefold()==province.casefold()]
            if area.empty: raise ValueError("Province not found in the boundary catalog")
            area=area.to_crs(4326)
            poly=area.geometry.union_all()
            pts=pts.to_crs(4326)
            pts=pts[pts.geometry.within(poly)].copy()
            lab["sample_id"]=lab.sample_id.astype(str)
            pts["sample_id"]=pts.sample_id.astype(str)
            if lab.sample_id.duplicated().any(): raise ValueError("Duplicate laboratory sample IDs")
            pts=pts.merge(lab[["sample_id",parameter]],on="sample_id",how="inner")
            pts[parameter]=pd.to_numeric(pts[parameter],errors="coerce")
            pts=pts.dropna(subset=[parameter])
            if len(pts)<8: raise ValueError("At least 8 matched numerical samples inside the province required")
            utm=pts.estimate_utm_crs()
            pts=pts.to_crs(utm);area=area.to_crs(utm)
            geom=area.geometry.union_all(); minx,miny,maxx,maxy=geom.bounds
            nx=int(np.ceil((maxx-minx)/resolution));ny=int(np.ceil((maxy-miny)/resolution))
            if nx*ny>350000: raise ValueError("Grid too large. Increase resolution")
            xs=minx+(np.arange(nx)+.5)*resolution
            ys=maxy-(np.arange(ny)+.5)*resolution
            ok=OrdinaryKriging(pts.geometry.x.to_numpy(),pts.geometry.y.to_numpy(),pts[parameter].to_numpy(),variogram_model="spherical",verbose=False,enable_plotting=False)
            zz,vv=ok.execute("grid",xs,ys[::-1])
            z=np.asarray(zz)[::-1,:].astype("float32");v=np.asarray(vv)[::-1,:].astype("float32")
            from rasterio.features import geometry_mask
            from rasterio.crs import CRS
            transform=from_origin(minx,maxy,resolution,resolution)
            mask=geometry_mask([mapping(geom)],out_shape=z.shape,transform=transform,invert=True)
            z[~mask]=-9999;v[~mask]=-9999
            import rasterio
            output=io.BytesIO()
            with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as out:
                for name,arr in [("prediction.tif",z),("variance.tif",v)]:
                    with MemoryFile() as mem:
                        with mem.open(driver="GTiff",height=ny,width=nx,count=1,dtype="float32",crs=utm,transform=transform,nodata=-9999,compress="deflate") as ds: ds.write(arr,1)
                        out.writestr(name,mem.read())
                out.writestr("municipalities.geojson",area.to_crs(4326).to_json())
                out.writestr("samples.geojson",pts.to_crs(4326).to_json())
                out.writestr("summary.json",json.dumps({"province":province,"parameter":parameter,"samples":len(pts),"resolution_m":resolution,"method":"ordinary_kriging","warning":"No cross-validation performed"}))
            output.seek(0)
            return StreamingResponse(output,media_type="application/zip",headers={"Content-Disposition":"attachment; filename=soil_fertility_outputs.zip"})
    except (ValueError,KeyError,zipfile.BadZipFile) as e: raise HTTPException(400,str(e))
