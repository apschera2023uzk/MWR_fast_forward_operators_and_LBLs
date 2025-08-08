# Evaluating Fast Radiative Transfer Models Using Ground-Based Microwave Radiometer Slant Path Observations at Low Elevations

This repository contains the code and auxiliary material used in the study:  
**"Evaluating Fast Radiative Transfer Models Using Ground-Based Microwave Radiometer Slant Path Observations at Low Elevations: RTTOV-gb and ARMS-gb"**

## Motivation

Slant path microwave observations (5–20° elevation) from ground-based radiometers offer a unique opportunity to test the performance of both line-by-line (LBL) and fast radiative transfer (RT) models under non-zenith geometries. This study aims to:

- Evaluate biases in fast RT models such as **RTTOV-gb** and **ARMS-gb**.
- Benchmark against **line-by-line radiative transfer models**, especially for water vapor channels.
- Analyze observational data from ground-based instruments and radiosondes.
- Quantify model performance under various assumptions of atmospheric homogeneity.

## How to setup the Code:

1. Download RTTOV-gb code from here: https://nwp-saf.eumetsat.int/site/software/rttov-gb/ and put it in your home directory ~/
2. Download git of pyrtlib: https://github.com/SatCloP/pyrtlib and put it also into your home directory ~/

## Scope of the Repository

This repository provides:

- Scripts to pre-process input from radiosondes and microwave radiometers.
- Interfaces for running:
  - **RTTOV-gb**
  - **ARMS-gb** (to be integrated)
  - **LBL models** (e.g. via mwrpy_ret or external)
- Tools for comparing modeled brightness temperatures with observations.
- Plotting and statistical analysis utilities.

## Use Case Highlights

1. **Bias evaluation of RTTOV-gb** retrievals
2. **Impact of slant path geometry** on RT model accuracy
3. **Comparison between fast models and LBL calculations**
4. Optional: **Integration of 3D water vapor fields**, based on high-res model output (ICON-D2)
5. Optional: **Forward model testing in a DA context** (e.g. OSSE setup)

## Data & Models Used

- Radiosonde measurements (Vital I, possibly Festival, Chile)
- Microwave radiometer TB elevation scans (JOYHAT / HAMHAT)
- Model input ICOn-D2
- Fast RT: **RTTOV-gb**, **ARMS-gb**
- LBL: TBD (likely LBLRTM via mwrpy_ret)

## Authors & Contributions

- **Main author:** Alexander Pschera
- **Co-authors:** None

## Structure (To be detailed)

> Note: This section will be filled as soon as the repository is structured.

## ⚠Requirements

See [`requirements.txt`](./requirements.txt) for environment setup.

## How to Run

> Will be documented once all scripts and configurations are complete.

## Planned Publication Repositories

- **Code**: GitHub (with Zenodo DOI on release)
- **Input/Output Data**:  
  - Option 1: Zenodo (for open access datasets)  
  - Option 2: Coscine (for institutional collaboration, FAIR metadata)
- **LBL Model Files**: Due to license restrictions, not included.

## Related Paper

- **Paper 1 (this repo)**: Focus on LBL & Fast RTM evaluation using slant path TBs

## License

> TBD — likely [MIT License](https://opensource.org/licenses/MIT)

---

Please cite this repository when using any part of the codebase in your own work. A citation file (`CITATION.cff`) will be added soon.

