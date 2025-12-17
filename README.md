# Evaluating Fast Radiative Transfer Models Using Ground-Based Microwave Radiometer Slant Path Observations at Low Elevations

This repository contains the code and auxiliary material used in the study:  
**"Evaluating Fast Radiative Transfer Models Using Ground-Based Microwave Radiometer Slant Path Observations at Low Elevations and LBL Models: RTTOV-gb and ARMS-gb"**

## Motivation

Slant path microwave observations (4–30° elevation) from ground-based radiometers offer a unique opportunity to test the performance of both line-by-line (LBL) and fast radiative transfer (RT) models under non-zenith geometries. This study aims to:

- Evaluate biases and random errors in fast RT models such as **RTTOV-gb** and **ARMS-gb**.
- Benchmark against **line-by-line radiative transfer models**, especially for water vapor channels.
- The addition of simulated observations from modle Goemetry in ICON-D2 is planned.

## How to setup the Code:

1. Download RTTOV-gb code from here: https://nwp-saf.eumetsat.int/site/software/rttov-gb/.
2. Download git of pyrtlib: https://github.com/SatCloP/pyrtlib.
3. Install ARMS-gb with FORTRAN intel compiler.
4. Install python packages from requirements.txt

## Scope of the Repository

This repository provides:

- Scripts to pre-process input from radiosondes and microwave radiometers.
- Interfaces for running:
  - **RTTOV-gb**
  - **ARMS-gb** (exact version not freely available yet)
  - **LBL models** (e.g. via PyRTlib)
- Tools for comparing modeled brightness temperatures with observations.
- Plotting and statistical analysis utilities.

## Data & Models Used

- Radiosonde measurements (compliation of the campaigns FESSTVaL, Socles and Vital I)
- Microwave radiometer TBs from azimuth scans, BL-scans and zenith for 5 HATPRO MWRs by RPG from these three campaigns)
- Model background or first guess fields will be added as inputs soon.

## Authors & Contributions

- **Main author:** Alexander Pschera

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

## Related Paper

- **Paper 1 (this repo)**: Once my paper on this topic is written it will be included here.

## License

> TBD — likely [MIT License](https://opensource.org/licenses/MIT)

---

Please cite this repository when using any part of the codebase in your own work. A citation file (`CITATION.cff`) will be added soon.

