# 8GEMSDOE Knowledge Base: Scientific Discovery of Geothermal Vents & Structural Controls

## 1. Geological Framework: Why Faults Host Geothermal Systems
In the Great Basin and the Walker Lane transtensional belt (Western Nevada), geothermal resources are almost exclusively **structurally controlled**. Unlike magmatic systems (such as the Cascades or Hawaii), the vast majority of geothermal activity in Nevada is non-magmatic: meteoric water circulates deeply along active fault fracture networks, is heated by the high regional geothermal gradient (~40–50 °C/km), and ascends rapidly along permeable fault conduits.

### Key Scientific Principles (Faulds & Hinz, 2011, 2012; Siler et al., 2022)
1. **Permeability Maintenance via Recurrent Slip**: Geothermal fluids are rich in dissolved silica, calcite, and trace minerals. As thermal fluids rise, decompress, and cool, mineral precipitation (self-sealing) rapidly seals fracture permeability within thousands of years. High permeability can only persist if active or recent tectonic stress continuously fractures the rock.
2. **Structural Geometry Dictates Fluid Upflow**: More than 85% of known commercial and high-temperature geothermal systems in the Great Basin do not occur along simple, planar, mid-segment fault traces. Instead, they cluster at specific geometric irregularities:
   - **Step-overs / Relay Ramps (En-echelon Normal Faults)**: Overlapping fault tips create localized pull-apart zones with high density of dilatational fractures (e.g. Steamboat Springs, Desert Peak, Brady, Tungsten Mountain).
   - **Fault Terminations / Tip Zones**: As slip tapers to zero at fault segment ends, displacement gradients induce intense splay fracturing and horse-tail damage zones.
   - **Fault Intersections**: Where dextral strike-slip faults of the Walker Lane intersect N-NNE striking normal faults of the Basin and Range, multiple fracture orientations intersect, maximizing vertical fracture connectivity.
   - **Accommodation Zones**: Broad belts where fault dip direction reversals occur (e.g. westward-dipping to eastward-dipping normal faults).

## 2. Geophysical & Remote Sensing Signatures of Hidden Geothermal Vents
Because most active geothermal systems are hidden beneath basin alluvium or pediment gravels ("blind systems"), surface geophysics are the primary discovery tool:
1. **Electrical Conductivity (`cond_surf`, Band 17)**:
   - Geothermal fluids alter volcanic and sedimentary host rocks into low-temperature hydrothermal clay assemblages—dominantly conductive smectite and illite.
   - A thick smectite clay cap forms above hydrothermal reservoirs, producing intense electrical conductivity anomalies.
2. **Geodetic Strain Rates (`geod_shearrate`, Band 7; `geod_dilaterate`, Band 8)**:
   - Active crustal deformation measured by GPS and InSAR. High shear strain rate combined with positive dilatation rate indicates active extensional opening, preventing mineral self-sealing and providing open conduits.
3. **Isostatic Gravity Anomaly Horizontal Gradient (`iso_grav_anom_hg`, Band 18)**:
   - Sharp horizontal gravity gradients mark the buried structural boundary between dense pre-Tertiary basement rock and low-density basin fill. Concealed range-front faults follow these steep gravity gradient ledges.
4. **Total Magnetic Intensity Derivatives (`tc`, Band 6; `tmi_hg`, Band 3)**:
   - Hydrothermal fluid circulation causes demagnetization (destroying magnetite into pyrite/hematite).
   - Tilt angle / total curvature (`tc`) serves as an edge-detection filter for structural boundaries and magnetic contrasts.
5. **Earthquake Seismicity (`ieq_n100a15`, Band 16; `deq_n100a15`, Band 10)**:
   - Micro-earthquake swarms cluster around active hydrothermal fluid injection and migration paths.

## 3. Official References & Scientific Verification Links
- **Faulds et al. (2011)**: *Structural Controls of Geothermal Activity in the Northern Basin and Range Province*, GRC Transactions: <https://publications.mygeoenergynow.org/grc/1029285.pdf>
- **USGS GeoDAWN Airborne Survey Release**: Northwestern Great Basin, Nevada and California, DOI: [10.5066/P93LGLVQ](https://doi.org/10.5066/P93LGLVQ)
- **INGENIOUS Project (GBCGE)**: Identification of High-Potential Geothermal Systems in the Great Basin: <https://gbcge.org/current-projects/ingenious/>
- **Siler et al. (2022)**: *Slip and Dilation Tendency on Great Basin Quaternary Faults*, USGS Data Release, DOI: [10.5066/P9YL58W6](https://doi.org/10.5066/P9YL58W6)
- **Peacock & Bedrosian (2022)**: *Electrical Conductance of the Great Basin*, USGS ScienceBase, DOI: [10.5066/P9TWT2LU](https://doi.org/10.5066/P9TWT2LU)
