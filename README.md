# ML Singapore Driver Drowsiness Detection

Our project aims to investigate the use of various machine learning models to detect driver drowsiness.

## Quickstart

1. Install dependencies using `pip install -r requirements.txt`
2. Use a virtual environment if you do not wish to pollute your global packages


## Video pre-processing

Run the script `process.py [source directory]` to process videos. The `source directory` should contain the videos
 to be processed placed in folders corresponding to each participant. 

This script extracts the video frames and applies a grayscale to them.

Adjust the `output_dir` variable at the start of the script to change the output directory for the generated frames.
 
Adjust the `rate` variable at the top of the script to change the number of frames to be extracted.