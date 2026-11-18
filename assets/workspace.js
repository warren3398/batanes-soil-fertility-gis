'use strict';

const $ = (id) => document.getElementById(id);
const form = $('mapping-form');
const province = $('province');
const api = location.hostname.endsWith('github.io') ? 'https://soil-fertility-gis-api.onrender.com' : '';
const views = [...document.querySelectorAll('[data-view]')];
const exports = [...document.querySelectorAll('[data-export]')];
const map = L.map('map', { zoomSnap: .25 }).setView([20.55, 121.91], 9);
const streets = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors', maxZoom: 19,
});
const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  attribution: 'Imagery &copy; Esri, Maxar, Earthstar Geographics and contributors', maxZoom: 19,
}).addTo(map);
L.control.layers({ Satellite: satellite, Streets: streets }).addTo(map);
L.control.scale({ imperial: false }).addTo(map);
const home = L.control({ position: 'topleft' });
home.onAdd = () => {
  const container = L.DomUtil.create('div', 'leaflet-bar');
  const button = L.DomUtil.create('button', 'map-home', container);
  button.type = 'button';
  button.title = 'Zoom to study area';
  button.setAttribute('aria-label', button.title);
  button.textContent = '⌂';
  L.DomEvent.disableClickPropagation(container);
  button.addEventListener('click', () => boundary.getBounds().isValid() ? fitBoundary() : map.setView([20.55, 121.91], 9));
  return container;
};
home.addTo(map);
const boundary = L.geoJSON(null, { style: { color: '#e4effc', weight: 2, fill: false } }).addTo(map);
const samples = L.geoJSON(null, {
  pointToLayer(feature, point) {
    return L.circleMarker(point, { radius: 4, weight: 1.3, color: '#182530', fillColor: '#fff5c5', fillOpacity: 1 });
  },
  onEachFeature(feature, layer) {
    const label = document.createElement('span');
    const data = feature.properties;
    label.textContent = [data.sample_id || data.id, run ? run.summary.parameter + ' ' + number(data[run.summary.parameter]) : ''].filter(Boolean).join(' · ');
    layer.bindTooltip(label);
  },
});
let run;
let catalogRequest;
const charts = new Map();
const overlays = new Map();
const number = (value, digits = 4) => Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—';

async function request(path, options) {
  const response = await fetch(api + path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((item) => item.msg).join(', ') : body.detail;
    throw new Error(detail || 'Server returned ' + response.status);
  }
  return response;
}

function fitBoundary() {
  const bounds = boundary.getBounds();
  if (bounds.isValid()) map.fitBounds(bounds, { padding: [14, 14], maxZoom: 16 });
}

async function showProvince() {
  catalogRequest?.abort();
  catalogRequest = new AbortController();
  const { signal } = catalogRequest;
  if (run || !province.value) return;
  try {
    const response = await request('/municipalities?province=' + encodeURIComponent(province.value), { signal });
    const area = await response.json();
    if (!signal.aborted && !run) {
      boundary.clearLayers().addData(area);
      fitBoundary();
    }
  } catch (error) {
    if (!signal.aborted) status('Boundary preview unavailable. ' + error.message, true);
  }
}

function status(message, error = false) {
  $('run-status').textContent = message;
  $('run-status').classList.toggle('error', error);
}

async function loadProvinces() {
  try {
    const response = await request('/provinces');
    const data = await response.json();
    province.replaceChildren(new Option('Select province', ''));
    for (const name of data.provinces) province.add(new Option(name, name));
    province.value = data.provinces.includes('Batanes') ? 'Batanes' : '';
    await showProvince();
  } catch (error) {
    province.replaceChildren(new Option('Catalog unavailable', ''));
    status('Province catalog unavailable. You can still upload a study boundary.', true);
  }
}
province.addEventListener('change', showProvince);

function swapChart(canvas, kind) {
  charts.get(canvas)?.destroy();
  const { validation, variogram, units } = run.summary;
  let datasets;
  let xTitle;
  let yTitle;
  if (kind === 'validation') {
    const points = validation.observed.map((x, i) => ({ x, y: validation.predicted[i] }));
    const values = [...validation.observed, ...validation.predicted];
    const min = Math.min(...values);
    const max = Math.max(...values);
    datasets = [
      { label: 'Out-of-fold prediction', data: points, backgroundColor: '#63b0ff', pointRadius: 3 },
      { label: '1:1 reference', data: [{ x: min, y: min }, { x: max, y: max }], showLine: true, pointRadius: 0, borderColor: '#f2887e', borderDash: [5, 4] },
    ];
    const unitLabel = units ? ' (' + units + ')' : '';
    xTitle = 'Observed' + unitLabel;
    yTitle = 'Predicted' + unitLabel;
  } else {
    datasets = [
      { label: 'Experimental', data: variogram.lag.map((x, i) => ({ x, y: variogram.gamma[i] })), backgroundColor: '#69befd', pointRadius: 3 },
      { label: 'Fitted ' + run.summary.model, data: variogram.distance.map((x, i) => ({ x, y: variogram.fitted[i] })),
        showLine: true, pointRadius: 0, borderColor: '#f29174', borderWidth: 2 },
    ];
    xTitle = 'Distance (m)';
    yTitle = 'Semivariance';
  }
  canvas.hidden = false;
  canvas.previousElementSibling.hidden = true;
  const axis = (text) => ({ title: { display: true, text, color: '#a6bad0' }, ticks: { color: '#b9cde0' }, grid: { color: '#ffffff17' } });
  charts.set(canvas, new Chart(canvas, {
    type: 'scatter', data: { datasets },
    options: { animation: false, maintainAspectRatio: false, scales: { x: axis(xTitle), y: axis(yTitle) },
      plugins: { legend: { labels: { color: '#e1eaf5', boxWidth: 11, font: { size: 10 } } } } },
  }));
}

function swatch(color, text) {
  const item = document.createElement('span');
  const chip = document.createElement('i');
  chip.className = 'swatch';
  chip.style.backgroundColor = color;
  item.append(chip, text);
  return item;
}

async function rasterLayer(zip, name, summary) {
  const filename = { prediction: 'map_classes.tif', variance: 'map_variance.tif', elevation: 'map_elevation.tif' }[name];
  const tif = await GeoTIFF.fromArrayBuffer(await zip.file(filename).async('arraybuffer'));
  const image = await tif.getImage();
  const [west, south, east, north] = image.getBoundingBox();
  const projection = L.CRS.EPSG3857;
  const bounds = L.latLngBounds(projection.unproject(L.point(west, south)), projection.unproject(L.point(east, north)));
  const width = image.getWidth();
  const height = image.getHeight();
  const pixels = await image.readRasters({ interleave: true });
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  const rgba = context.createImageData(width, height);
  const palette = summary.classes.map(({ color }) => color.match(/[a-f\d]{2}/gi).map((byte) => parseInt(byte, 16)));
  const range = summary[name];
  for (let i = 0; i < pixels.length; i++) {
    const value = pixels[i];
    if (name === 'prediction' ? !value : value === -9999 || !Number.isFinite(value)) continue;
    const fraction = range ? Math.max(0, Math.min(1, (value - range[0]) / (range[1] - range[0] || 1))) : 0;
    const rgb = name === 'prediction' ? palette[value - 1] :
      name === 'variance' ? [35 + 211 * fraction, 158 - 65 * fraction, 231 - 182 * fraction] :
        [62 + 181 * fraction, 83 + 152 * fraction, 73 + 145 * fraction];
    rgba.data.set([...rgb, 224], i * 4);
  }
  context.putImageData(rgba, 0, 0);
  return L.imageOverlay(canvas.toDataURL(), bounds, { opacity: .95, className: 'raster' });
}

function displayView(next) {
  for (const button of views) button.setAttribute('aria-pressed', String(button.dataset.view === next));
  const chart = next === 'variogram' || next === 'validation';
  const chartPanel = $('large-chart');
  chartPanel.hidden = !chart;
  $('map').parentElement.hidden = chart;
  if (chart) {
    if (run) swapChart(chartPanel.querySelector('canvas'), next);
    return;
  }
  map.invalidateSize();
  for (const [name, overlay] of overlays) {
    if (name === next) overlay.addTo(map);
    else overlay.remove();
  }
  if (run && (next === 'samples' || next === 'prediction')) samples.addTo(map).bringToFront();
  else samples.remove();
  boundary.bringToFront();
  const legend = $('legend');
  legend.hidden = !run || next === 'samples';
  if (!run || next === 'samples') return;
  const { summary } = run;
  const heading = document.createElement('strong');
  heading.textContent = next === 'prediction' ? summary.parameter + ' · ' + (summary.units || 'Source units') :
    next === 'variance' ? 'Kriging variance' : 'Elevation';
  legend.replaceChildren(heading);
  if (next === 'prediction') {
    for (const row of summary.classes) {
      const line = document.createElement('div');
      line.append(swatch(row.color, row.range + ' · ' + row.label));
      legend.append(line);
    }
  } else {
    const line = document.createElement('div');
    line.textContent = summary[next].map((value) => number(value)).join(' – ') + (next === 'variance' && summary.units ? ' ' + summary.units + '²' : '');
    const ramp = document.createElement('div');
    ramp.className = 'legend-ramp ' + next;
    legend.append(ramp, line);
  }
}
for (const button of views) button.addEventListener('click', () => displayView(button.dataset.view));

function renderResults(area, points) {
  const info = run.summary;
  boundary.clearLayers().addData(area);
  samples.clearLayers().addData(points);
  const rows = info.classes.map((item) => [swatch(item.color, item.label), item.range, number(item.area_ha, 2), number(item.percent, 2)]);
  rows.push(['Total', '', number(info.classes.reduce((sum, item) => sum + item.area_ha, 0)), '100']);
  $('classes').replaceChildren(...rows.map((values) => {
    const row = document.createElement('tr');
    for (const value of values) row.insertCell().append(value);
    return row;
  }));
  $('class-note').textContent = info.classification + ' · ' + (info.units || 'Source units');
  for (const metric of ['r2', 'rmse', 'mae', 'mbe']) $(metric).textContent = number(info.validation[metric]);
  swapChart($('scatter'), 'validation');
  swapChart($('variogram'), 'variogram');
  views.find((button) => button.dataset.view === 'elevation').hidden = !info.elevation;
  for (const button of exports) button.disabled = false;
  displayView('prediction');
  fitBoundary();
  $('run-meta').textContent = [info.province, info.crs, info.resolution_m + ' m', info.samples + ' samples', info.model].join(' · ');
}

const autoFit = $('auto-fit');
const modelSelect = form.elements.model;
function manualSettings() {
  const model = modelSelect.value.toLowerCase();
  $('manual-label').textContent = model === 'linear' ? 'Slope, nugget' :
    model === 'power' ? 'Scale, exponent, nugget' : 'Full sill, range (m), nugget';
  $('manual-field').hidden = autoFit.checked;
  form.elements.manual_variogram.disabled = autoFit.checked;
  form.elements.manual_variogram.required = !autoFit.checked;
}
autoFit.addEventListener('change', manualSettings);
modelSelect.addEventListener('change', manualSettings);
manualSettings();

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  if (data.get('boundary').size) data.delete('province');
  if (!data.get('province') && !data.get('boundary').size) {
    status('Choose a province or upload a study boundary.', true);
    return;
  }
  for (const name of ['boundary', 'dem', 'results']) if (!data.get(name).size) data.delete(name);
  data.set('model', data.get('model').toLowerCase());
  data.set('dashboard', 'true');
  catalogRequest?.abort();
  const submit = form.querySelector('[type=submit]');
  const started = Date.now();
  submit.disabled = true;
  $('progress').hidden = false;
  $('footer-status').textContent = 'Processing';
  status('Validating inputs, fitting the variogram, and running kriging and cross-validation…');
  try {
    const response = await request('/interpolate', { method: 'POST', body: data });
    status('Preparing map layers and reports…');
    const blob = await response.blob();
    const zip = await JSZip.loadAsync(blob);
    const read = async (filename) => JSON.parse(await zip.file(filename).async('string'));
    const [summary, area, points] = await Promise.all([read('summary.json'), read('municipalities.geojson'), read('samples.geojson')]);
    const names = summary.elevation ? ['prediction', 'variance', 'elevation'] : ['prediction', 'variance'];
    const layers = await Promise.all(names.map((name) => rasterLayer(zip, name, summary)));
    for (const overlay of overlays.values()) overlay.remove();
    overlays.clear();
    names.forEach((name, i) => overlays.set(name, layers[i]));
    run = { blob, zip, summary };
    renderResults(area, points);
    status('Mapping complete. GIS layers and reports are ready.');
    $('footer-status').textContent = 'Complete';
    $('timer').textContent = ((Date.now() - started) / 1000).toFixed(1) + ' s';
  } catch (error) {
    status(error.message, true);
    $('footer-status').textContent = 'Run failed';
  } finally {
    submit.disabled = false;
    $('progress').hidden = true;
  }
});

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const rasters = ['prediction.tif', 'variance.tif', 'classification.tif'];
for (const button of exports) {
  button.addEventListener('click', async () => {
    if (!run) return;
    const name = button.dataset.export;
    try {
      if (name === 'all') {
        download(run.blob, 'soil_fertility_outputs.zip');
      } else if (name === 'rasters' || name === 'qgis') {
        const source = run.zip;
        const files = name === 'qgis' ? [...rasters, 'soil_layers.gpkg', 'soil_fertility.qgz'] : rasters;
        const archive = new JSZip();
        await Promise.all(files.map(async (file) => archive.file(file, await source.file(file).async('uint8array'))));
        download(await archive.generateAsync({ type: 'blob' }), name === 'qgis' ? 'soil_qgis_project.zip' : 'soil_geotiffs.zip');
      } else {
        download(await run.zip.file(name).async('blob'), name);
      }
    } catch (error) {
      status('Export failed. ' + error.message, true);
    }
  });
}
loadProvinces();
