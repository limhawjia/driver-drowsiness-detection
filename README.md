# ML Singapore Driver Drowsiness Detection

Our project aims to investigate the use of various machine learning models to detect driver drowsiness.

## Quickstart

1. Install dependencies using `pip install -r requirements.txt`
2. Use a virtual environment if you do not wish to pollute your global packages


## Video pre-processing

Run the script `process.py` to process videos. This script extracts the video frames and applies a grayscale to them.

Adjust the variables `video_dir` and `output_dir` variables at the start of the script to change the source directory
 for the videos and the output directory for the generated frames. 

Adjust the `rate` variable at the top of the script to change the number of frames to be extracted.