# TEAMUP IV Curve Analysis Project

## Overview
This project provides a complete pipeline for analyzing photovoltaic (PV) IV curves, extracting key device parameters, and calculating series resistance using the Sites method. It supports both single-file and batch analysis, generates publication-quality plots, and exports results to CSV.

## Features
- **IV Curve Analysis**: Extracts Voc, Jsc, Fill Factor, Maximum Power Point, and more from IV data.
- **Sites Method**: Calculates series resistance (Rs) and ideality factor (n) using the Sites method.
- **Data Cleaning**: Outlier removal and data smoothing.
- **Batch Processing**: Analyze multiple measurements from Excel files.
- **Visualization**: Generates comprehensive plots for IV curves, power curves, differential resistance, and Sites method fits.
- **Results Export**: Saves analysis results and plots to the `analysis_results/` folder.

## Folder Structure
```
TEAMUP-DARSH/
├── Ch102_2025_05_18wCalculations.xlsx   # Example raw data file
├── data_loader.py                       # Loads and parses PV data from Excel
├── ivcurve.py                           # Main all-in-one IV analysis script
├── series_resistance_calculator.py      # Standalone Sites method calculator
├── IV_graph.py                          # (Empty or for custom plotting)
├── analysis_results/                    # Output folder for results
│   ├── analysis_results.csv             # CSV of analysis results
│   └── sites_analysis_complete.png      # Summary plot
├── data/                                # Example/sample data
│   └── sample_iv_curve.xlsx             # Example IV curve data
```

## Main Scripts
### 1. `ivcurve.py`
- **Purpose**: All-in-one script for IV curve analysis and Sites method.
- **Usage**:
  ```sh
  python ivcurve.py <path_to_excel> [area_cm2]
  ```
  If no file is provided, it generates and analyzes sample data.
- **Output**: Plots and CSV in `analysis_results/`.

### 2. `series_resistance_calculator.py`
- **Purpose**: Standalone Sites method calculator for batch data (e.g., from `Ch102_2025_05_18wCalculations.xlsx`).
- **Usage**:
  ```sh
  python series_resistance_calculator.py
  ```
  (Edit the script to set your input file.)

### 3. `data_loader.py`
- **Purpose**: Utility for loading and parsing IV curve data from Excel files.

### 4. `IV_graph.py`
- **Purpose**: (Currently empty or for custom plotting.)

## Input Data
- **Format**: Excel files with columns for voltage/current or stringified IV curve arrays.
- **Example**: See `data/sample_iv_curve.xlsx` and `Ch102_2025_05_18wCalculations.xlsx`.

## Output
- **Plots**: Saved as PNG in `analysis_results/`.
- **Results**: CSV summary in `analysis_results/analysis_results.csv`.

## Requirements
- Python 3.8+
- Packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `openpyxl`

Install dependencies with:
```sh
pip install numpy pandas matplotlib scipy openpyxl
```

## References
- Sites, J. R. (Notes on Series Resistance Extraction)
- [Pyscha et al., 2020]

## License
MIT License

---
*For questions or contributions, please contact the project maintainer.*
