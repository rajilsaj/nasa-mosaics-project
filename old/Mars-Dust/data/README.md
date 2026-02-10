# VERSA Data Overview
VERSA: Vortex Event Reactive Sensor Algorithm

## sol_*
Directories containing MEDA data downloaded from the PDS

## compiled_meda.csv
Single CSV containing all PDS data compiled into one file.

## Jackson_vortex_detections*
Hand labeled vortex information from the Jackson, 2022 paper. The original information from the computer-ready table is in `Jackson_vortex_detections.txt`. A processed, better-formatted version (ready for loading via pandas) is in `Jackson_vortex_detections_reformatted.txt`

The CSV with corresponding MEDA and spacecraft clock (SCLK) times (`Jackson_vortex_dtections_reformatted_augmented.csv`) is most useful for evaluation and was used to generate the ML-ready data.

## ml_ready_vortex_data.csv
ML-Ready CSV useful for training vortex detectors. Columns include
* SCLK - spacecraft clock time useful as an index
* Pressure - pressure reading at this timepiont
* sol - Mars sol
* time - Mars time of day
* gt_fwhm - Ground truth, full-width half max window. Whether or not this time point lies within the FWHM window of a vortex. This period contains the critical science we want to capture
* gt_vortex_ind - index of the vortex referring back to the vortices labeled in Jackson, 2022
* gt_detection_win - Whether or not this timepoint is within the 60 seconds directly preceeding a FWHM window. The exact 60-second window timing is somewhat arbitrary, but we want to train detectors to trigger within a short window BEFORE the FWHM window begins.
* gt_4xfwhm - Ground truth, 4 times the FWHM window. Whether or not this time point is within +/- 2 FWHM windows of the vortex center. Useful for training detectors to identify if a current timepoint is part of a vortex. The original VERSA team used this column to train ML models.
* PRESSURE_MA_500 - Convience column containing the simple 500-sample moving average. Useful for subtracting from the PRESSURE column to get a pressure difference signal as an input to ML or other models. This is a good input column to use for modeling as the pressure on Mars drifts substantially over each sol and over the seasons since CO2 freezes out at the caps.
