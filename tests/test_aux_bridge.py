"""The aux-channel git bridge must round-trip within its stated error bound and refuse bad bytes."""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ab = _load("aux_bridge")

H, W = 60, 80
TRANSFORM = from_origin(243350.0, 4508550.0, 100.0, 100.0)


def _template(path: Path) -> np.ndarray:
    a = np.zeros((H, W), np.float32)
    a[:, :10] = np.nan                       # outside the footprint
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                       crs="EPSG:32611", transform=TRANSFORM, nodata=np.nan) as d:
        d.write(a, 1)
    return np.isfinite(a)


def _source(path: Path, bands: np.ndarray, transform=TRANSFORM):
    with rasterio.open(path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2],
                       count=bands.shape[0], dtype="float32", crs="EPSG:32611", transform=transform,
                       nodata=np.nan) as d:
        for i in range(bands.shape[0]):
            d.write(bands[i], i + 1)
            d.update_tags(i + 1, band_name=f"b{i}")


def test_roundtrip_within_bound_multi_part(tmp_path):
    foot = _template(tmp_path / "t.tif")
    rng = np.random.default_rng(0)
    src = np.stack([rng.normal(10, 3, (H, W)), rng.uniform(-5, 5, (H, W)), np.linspace(0, 1, H * W).reshape(H, W)]
                   ).astype(np.float32)
    src[1, 5:8, 20:25] = np.nan              # a hole inside the footprint
    _source(tmp_path / "s.tif", src)
    m = ab.pack(tmp_path / "s.tif", "x", tmp_path / "t.tif", bridge_root=tmp_path / "br", part_bytes=4096)
    assert len(m["parts"]) >= 2, "small part size must force a split"
    assert [b["band_name"] for b in m["bands"]] == ["b0", "b1", "b2"]
    out = ab.unpack("x", tmp_path / "o.tif", tmp_path / "t.tif", bridge_root=tmp_path / "br")
    with rasterio.open(out) as o:
        assert o.transform == TRANSFORM and o.count == 3 and o.dtypes[0] == "float32"
        for i in range(3):
            got = o.read(i + 1)
            lo, hi = m["bands"][i]["lo"], m["bands"][i]["hi"]
            inside = foot & np.isfinite(src[i]) & (src[i] >= lo) & (src[i] <= hi)
            err = np.abs(got[inside] - src[i][inside])
            assert err.max() <= (hi - lo) / 508 + 1e-5
            assert np.isnan(got[~foot]).all(), "outside the footprint must be NaN"
            assert o.tags(i + 1)["band_name"] == f"b{i}"
        assert np.isnan(o.read(2)[5:8, 20:25]).all(), "an in-footprint hole must stay a hole"


def test_tampered_part_is_refused(tmp_path):
    _template(tmp_path / "t.tif")
    _source(tmp_path / "s.tif", np.ones((1, H, W), np.float32))
    ab.pack(tmp_path / "s.tif", "x", tmp_path / "t.tif", bridge_root=tmp_path / "br", part_bytes=2048)
    p = sorted((tmp_path / "br" / "x").glob("*.part-*"))[0]
    b = bytearray(p.read_bytes())
    b[100] ^= 0xFF
    p.write_bytes(bytes(b))
    with pytest.raises(SystemExit):
        ab.verify("x", bridge_root=tmp_path / "br")


def test_grid_mismatch_is_refused(tmp_path):
    _template(tmp_path / "t.tif")
    _source(tmp_path / "s.tif", np.ones((1, H, W), np.float32), transform=from_origin(243450.0, 4508550.0, 100, 100))
    with pytest.raises(SystemExit):
        ab.pack(tmp_path / "s.tif", "x", tmp_path / "t.tif", bridge_root=tmp_path / "br")


def test_radiometric_builder_on_synthetic_zips(tmp_path, monkeypatch):
    """End to end: two zips with *_k_a2 / *_tc_a2 / *_k_a1 grids -> mosaic on the template grid."""
    rb = _load("build_radiometric_channel")
    foot = _template(tmp_path / "t.tif")
    # official-features stand-in: 6 bands, band 6 = the same field as the synthetic total count
    yy, xx = np.mgrid[0:H, 0:W]
    tc = (10 + 0.2 * xx + 0.1 * yy).astype(np.float32)
    feats = np.stack([np.zeros((H, W), np.float32)] * 5 + [tc])
    _source(tmp_path / "f.tif", feats)
    work = tmp_path / "work"
    work.mkdir()
    # area 2 covers the east half at 100 m; area 1 covers everything at 50 m (fills the west)
    a2 = work / "a2"
    a1 = work / "a1"
    a2.mkdir()
    a1.mkdir()
    k2 = np.where(xx >= W // 2, 2.0, np.nan).astype(np.float32)
    tc2 = np.where(xx >= W // 2, tc, np.nan).astype(np.float32)
    _source(a2 / "22103_k_a2.tif", k2[None])
    _source(a2 / "22103_tc_a2.tif", tc2[None])
    k1 = np.full((2 * H, 2 * W), 1.0, np.float32)
    _source(a1 / "22103_k_a1.tif", k1[None], transform=from_origin(243350.0, 4508550.0, 50, 50))
    tc1 = np.repeat(np.repeat(tc, 2, 0), 2, 1)
    _source(a1 / "22103_tc_a1.tif", tc1[None], transform=from_origin(243350.0, 4508550.0, 50, 50))
    zdir = tmp_path / "zips"
    zdir.mkdir()
    for zname, d in (("22103_area1_tiffs.zip", a1), ("22103_area2_tiffs.zip", a2)):
        with zipfile.ZipFile(zdir / zname, "w") as z:
            for f in d.iterdir():
                z.write(f, f.name)
        rb.ZIPS[zname]["md5"] = rb.md5_file(zdir / zname)
        rb.ZIPS[zname]["bytes"] = (zdir / zname).stat().st_size
    out, rep = tmp_path / "rad.tif", tmp_path / "rep.json"
    monkeypatch.setattr(sys, "argv", ["x", "--zip-dir", str(zdir), "--template", str(tmp_path / "t.tif"),
                                      "--features", str(tmp_path / "f.tif"), "--out", str(out), "--report", str(rep)])
    assert rb.main() == 0
    r = json.loads(rep.read_text())
    assert r["band_names"] == ["rad_k", "rad_tc"]
    with rasterio.open(out) as o:
        k = o.read(1)
    assert np.allclose(k[foot & (xx >= W // 2 + 1)], 2.0), "area 2 is the base"
    assert np.allclose(k[foot & (xx < W // 2 - 1)], 1.0), "area 1 fills where area 2 is empty"
    assert np.isnan(k[~foot]).all()
    t6 = r["band6_identity_test"]
    assert t6["spearman"] > 0.99 and "IS the GeoDAWN" in t6["verdict"]
