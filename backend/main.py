import io, json, tempfile, zipfile, os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pykrige.ok import OrdinaryKriging
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import mapping
app=FastAPI(title="Soil Fertility GIS API")
app.add_middleware(CORSMiddleware,allow_origins=["https://warren3398.github.io","http://localhost:8000"],allow_methods=["*"],allow_headers=["*"])
@app.get("/health")
def health(): return {"status":"ok","method":"ordinary_kriging","boundaries":"upload_required"}
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
@app.post("/interpolate")
async def interpolate(points:UploadFile=File(...),results:UploadFile=File(...),boundaries:UploadFile=File(...),province:str=Form(...),parameter:str=Form("N"),resolution:int=Form(100)):
    if resolution<10 or resolution>5000: raise HTTPException(400,"Resolution must be 10-5000 m")
    try:
        with tempfile.TemporaryDirectory() as d:
            pts=loadshape(await points.read(),Path(d)/"points")
            bounds=loadshape(await boundaries.read(),Path(d)/"boundaries")
            lab=pd.read_csv(io.BytesIO(await results.read()))
            for df,cols in [(pts,["sample_id"]),(lab,["sample_id",parameter]),(bounds,["province","municipality"])]:
                missing=[x for x in cols if x not in df.columns]
                if missing: raise ValueError("Missing fields: "+", ".join(missing))
            if pts.geometry.geom_type.ne("Point").any(): raise ValueError("Sampling layer must contain points")
            area=bounds[bounds.province.astype(str).str.casefold()==province.casefold()]
            if area.empty: raise ValueError("Province not found in uploaded boundaries")
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
