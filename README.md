# Compressive Streak Microscopy
We developed a compressive streak fluorescence microscope tailored to record sparse fluorescence imaging data at sampling speed greater than 200Hz.

The galvo mirror, the camera external trigger, DMD (usually a fixed pattern, changing pattern only for beads experiment), LED, red LED (if available) were synchronized through a NiDAQ controlled by MATLAB. Camera acquisition was controlled by Micromanager through Python. Data processing files were written in Python. We tested different reconstruction algorithms including ridge regression, weighted averaging and non-negative matrix factorization.

![Fig1_v1 1-01](https://github.com/user-attachments/assets/3f3bd9a5-34ff-403f-a388-9c73fa6d0076)
