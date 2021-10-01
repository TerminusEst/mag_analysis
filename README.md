# mag_analysis
Routines for analysing simulated and historical geomagnetic storm data with Python

## GroundMag_functions.py

- Functions to read in SuperMag and INTERMAGNET data from local files. 
- Dst and Sym-H read in also
- E-field calculation for a 1-D ground resistivity model.
- Auroral boundary calculator

## SWMF_functions.py

- Function to read in and BATSRUS mag-grid files and save as numpy objects
- Quickly read in and plot SWMF solar wind conditions from IMF.dat file
- Calculate magnetopause standoff distance for BATSRUS y=0 2D magnetosphere files
- Plot azimuthal J in near Earth magnetosphere for different planes around Earth

