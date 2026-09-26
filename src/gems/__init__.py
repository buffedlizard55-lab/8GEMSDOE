"""8GEMSDOE — GEMS Prize strategy line 8.

Modules:
  metric     distance-weighted Tversky index (official formulas) + known-fault masking
  features   potential-field / topographic / strain transforms (numpy+scipy only)
  submission GeoTIFF build + 13-gate validation (tifffile, no GDAL required)
  holdout    spatially-blocked, buffered folds + proxy scoring harness
"""

__version__ = "0.1.0"
