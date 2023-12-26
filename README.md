# Panorama Construction Project

This project is focused on constructing panoramic images from a set of input images. The code uses computer vision techniques, including keypoint extraction, feature matching, and homography estimation, to stitch together multiple images into a seamless panorama.

## Usage

To run the code, follow the steps below:

1. Clone the repository:

   ```bash
   git clone https://github.com/Chandanagowda23/Panorama-Construction.git
   cd Panorama-Construction

2. Run the script:

   python task2.py --input_path data/images_panaroma --output_overlap ./task2_overlap.txt --output_panaroma ./task2_result.png

3. Command Line Arguments:
--input_path: Path to the directory containing images for panorama construction.
--output_overlap: Path to save the overlap result in JSON format.
--output_panaroma: Path to save the final panorama image.

## Code Structure

task2.py: The main script responsible for panorama construction.
utils.py: Contains utility functions used in the panorama construction process.

## Dependencies
OpenCV
NumPy
Matplotlib

### Install the required dependencies using:
pip install opencv-python numpy matplotlib

## Workflow

>Image Loading and KeyPoint Extraction:
Images are loaded from the specified path, and keypoint features are extracted using the Scale-Invariant Feature Transform (SIFT).
>Overlap Calculation:
An NxN one-hot array is generated, indicating the pairwise overlap between images. Overlapping regions are determined based on feature matching.
>Image Stitching:
The code iteratively finds the best pair of images to stitch together, considering homography and feature matching scores. The stitching process removes black padding to create a visually appealing panorama.
>Saving Results:
The final panorama image is saved as task2_result.png, and the overlap matrix is saved as task2_overlap.txt in JSON format.
Output

The final panorama image: task2_result.png.
Overlap matrix (JSON): task2_overlap.txt.


