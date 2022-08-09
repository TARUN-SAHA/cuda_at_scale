# CUDA at scale project

This is modified version of sample from Nvidia CUDA toolkit.
This code demonstrates Canny edge detection filter. Input is a color image(s) and output is 8-bit grayscale image(s) highlighting edges in the image.
Sample input images are taken from tensorflow cat_vs_dogs dataset (https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)

Directory structure:
src/ : cpp source file for this project
lib/ : header files required for this project
data/images: input images on which tests were run
data/output: output images after executing the test
bin/ : generated executable will be in this directory

Build steps:
* git clone https://github.com/TARUN-SAHA/cuda_at_scale.git
* cd cuda_at_scale
* make clean build

Usage: Output will be saved in data/output directory with same as input image filename.
# default execution (process all images in data/images directory)
./bin/edgeDetector (or 'make run')
# process a single image
./bin/edgeDetector -input ./<image_path>
# process all files in directory
./bin/edgeDetector -input ./<image directory>

# Image format
Input Image format: JPEG
Output Image format: JPEG

