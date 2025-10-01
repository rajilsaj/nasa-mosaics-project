# NASA–MOSAICS / ISL — Cassini ELS & Mars Dust Pipeline (Private)

> **Status:** Internal research project (ISL × NASA/JPL).
> **Distribution:** **Private** — not open source. Do not share or redistribute without written approval.

## 🔗 Quick Links

* **Part I — Mars Dust Collection (internal):** 👉 **[Mars Dust](./Mars-Dust/)
* **Part II — Cassini ELS (GitHub folder):** 👉 **[Cassini ELS](./Cassini-ELS/)

---

## Overview

This repository hosts the **end-to-end data workflow** we use for:

1. **Mars Dust Collection (Part I):** Internal acquisition, curation, and labeling of Martian dust-devil observations for downstream ML (ISL/NASA). This project, also known as VERSA (Vortex Event Reactive Sensor Algorithm), employs a power-efficient two-stage LSTM pipeline to detect Martian dust vortices using pressure sensor data. The first stage uses an LSTM Autoencoder to filter normal data, while the second stage uses an LSTM Classifier to detect vortices in the filtered data.

2. **Cassini CAPS/ELS context (Part II):** Mirroring and parsing Cassini **Electron Spectrometer (ELS)** products (plus Zenodo boundary labels) for plasma context and validation.

---

## Repository Layout

```
/Mars-Dust/                  # Part I (internal) - VERSA project
  /data/                     # MEDA data, hand-labeled vortex info, and ML-ready data
  /models/                   # LSTM and prediction models
  /utils/                    # Utility scripts for analysis and visualization
  /Anita/                    # Additional scripts
  requirements.txt           # Python dependencies for the VERSA project
/Cassini-ELS/                # Part II - Cassini ELS analysis
/README.md                   # This file
```

---

## Minimal Workflow (Cassini ELS)

```bash
# Mirror ELS (preserve structure)
python3 scripts/simple_mirror.py \
  https://pds-ppi.igpp.ucla.edu/data/cassini-caps-calibrated/data-els/ \
  raw/cassini_caps_els

# Parse .DAT + .xml ➜ .npz (mirrored under /processed)
python3 scripts/build_caps_products.py
```

**NPZ contents (per product):**

* `data` → (nrec, 63 energies, 8 anodes) counts/s
* `energies_eV`, `theta_deg`, `phi_deg`
* `utc`, `dt`, telemetry/dead_time flags
* Spacecraft geometry & rotations (J2000, RTP)

> Converting counts/s to flux requires instrument response & potential corrections (outside this repo).

---

## Environment

```bash
# For the VERSA (Mars-Dust) project
pip install -r Mars-Dust/requirements.txt

# For Cassini ELS analysis
conda create -n caps python=3.11 numpy
conda activate caps
# add matplotlib/pandas/jupyter if you’ll use notebooks/plots
```

---

## Data Rights & Attribution

* **Zenodo dataset (boundary labels):** **CC BY 4.0** — cite:
Jackman, C., Thomson, M., Dougherty, M., & Daigavane, A. (2021). *Magnetic Field Boundaries in Cassini Plasma Spectrometer Data* (1.0.4) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.5004160](https://doi.org/10.5281/zenodo.5004160)
* Acknowledge the **PDS Planetary Plasma Interactions (PPI) Node** when using ELS data.
* **Part I (Mars Dust Collection):** internal to ISL/NASA collaborators. **Do not** redistribute.

---

## Contributors

* **PI / Mentor(s):** [add names/affiliations]
* **ISL Maintainer:** [add contact]
* **Data Questions (ELS/PDS):** PDS PPI Node docs / CAPS user guide (internal notes)

---

## Compliance

This repository may reference public resources, but the **assembled workflow, curation, and annotations are private**. Ensure all shares/publications comply with ISL/NASA agreements and dataset licenses.