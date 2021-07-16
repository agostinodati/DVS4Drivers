# DVS4Drivers
Computer Vision &amp; Deep learning project

This Python Project aims to test landmarks estimation through optical flow on images reconstructed from events recorded from an event camera.

The two videos used in the project can be found at:
- https://drive.google.com/file/d/1vgkQXB5xcFtAB5jegYpDmycKRUEzz3ns/view?usp=sharing
- https://drive.google.com/file/d/1wXtL2T8Gz7V6iQ9QwiJU1DnArsieuDqA/view?usp=sharing

## Requirements:
- opencv
- numpy
- dv
- matplotlib
- mediapipe


## main.py
In this file are located the main function of the project:
- main_optical_flow_naive(path, timeskip) : This function tests the optical flow with the naive representation of the events.
- main_optical_flow_accumulator(path, timeskip): This function tests the optical flow with the accumulator representation of the events.
- main_blink_mouth(filepath, timeskip) : This prototype function aims to detect when the eyes blink and when the mouth is openened/closed. 

All the functions require the path of the aedat4 file. The timeskip is optional (default is zero).

How to use: in the main section, type the function that you want to use and pass the file path.

## dvs4d_lib.py
In this file are located all the auxiliary functions used by the main's function.
  
## landmark_indexes.py
In this file are located the Facemesh's landmarks needed for the project (face silhouette, eyes and lips).
 
## errorplots.zip
This zipped file contains the error graphs plotted during the tests.
  
## DVS4Driver_results.xlsx
This xlsx file cointains the raw values of the tests.
