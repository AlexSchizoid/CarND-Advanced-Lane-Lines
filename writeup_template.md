## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/result_undistorted_calibration1.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/result_undistorted_test1.jpg "Binary Example"
[image4]: ./output_images/result_warped_test1.jpg "Warp Example"
[image5]: ./output_images/result_threshold_test1.jpg "Color Example"
[image6]: ./output_images/result_threshold_test2.jpg "Color Example"
[image7]: ./output_images/result_threshold_test3.jpg "Color Example"
[image8]: ./examples/result_lanes_test1.jpg "Fit Visual Output"
[image9]: ./examples/result_lanes_test2.jpg "Fit Visual Output"
[image10]: ./examples/result_lanes_test3.jpg "Fit Visual Output"
[video11]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in cells 2,3,4,5 of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

The distortion correction is applied using OpenCV cv2.undistort and the calibration parameters we obtain earlier - undistort_image function.

![alt text][image3]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_perspective()`, which appears in cell 2,7 and 8.  The `warp_perspective()` function takes as inputs an image (`img`), as well as a transormation matrix M  which is used by OpenCv to warp the image. The M matrix is obtained from source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

The following is an example from this stage of the pipeline. You can see the original image and the 'birds-eye view':

![alt text][image4]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color thresholds to generate a binary image. 
To isolate the yellow lane I used the B channel of the LAB color space. The B channel is also called the blue–yellow channel, and a threshold of (145, 255) seemed to work well for the video and test_images. 
To isolate the white lane I used the Yc channel of the YCbCr color space, which also called luminance. It managed to isolated the white lane pretty well in the test images and videos, across diferrent lighting conditions. I used a threshold of  (220, 255)

I also tried to implement a directional and magnitude sobel operator but i found that in my case it mostly added noise to output.

The code can be found in functions apply_all_thresholds, b_threshold, yc_thresold.
Here are some examples of my output for this step. The output images are binary images.

![alt text][image5]
![alt text][image6]
![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to fit lane-lines on top of the warped perspective I first applied a histogram on the previous layer images. Taking the two peaks which corespond to the lane lines are very good starting points for the x base of the lanes. This base is a good place to start lookging for points - we check for points at an offset from the center - the margin. Afterwards i proceeded with a sliding window search all the way to the top of the image, checking which points are members of the line. This is done for both left and right lines. After we obtain these points we can proceed to fit a second order polynomial on both sets of points. 

I implemented this step in the function find_line.
Here are some outputs from this stage. On the left is the birds-eye image with the lines-fitted, while on the right is the final output of the pipeline which is the unwarped image with the lanes being correctly identified.

![alt text][image8]
![alt text][image9]
![alt text][image10]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature and position of the vehicle are calculated in the find_line function and outputed on the final image.
For the radius we used the formula learned in the courses, and involves taking the first and second derivatives of the polynomials that we fit on the lines. 

We can assume the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines we've detected. The offset of the lane center from the center of the image (converted from pixels to meters) is the distance from the center of the lane. 

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The video pipeline uses a lot of the stages I discussed previously.
-we undistort the image
-we warp it
-we apply color thresholds
-we try to obtain lane points and fit lines
    -if no previous lines were found we do a full sliding window search i discussed previously
    -if we have previous points found than we can just check around a margin of those points for a fast search.
        - this works like a look ahead filter
        - if not enough points are found we can do a fast search
    -i try to smooth the lanes by averagin line fits from the last 5 frames
    -if both methods don't find lanes in the current frame than i just use the previous best fit
 -use the points and lines obtained to calculate radius and offset
 -output image to video

Here's a [link to my video result](./result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The implementation works quite well for the project video. Because the color/gradient thresholds are hardcoded i'm expecting the pipeline to have problems when there are dramatic differences in lighting and lane curvature change. Instead of static threshold i feel there is a need for an adaptive threshold that takes information from previous frames in order to provide better intermediary outputs. This is very important in order to obtain good fitted lines. Also I feel that I need some better sanity checks in order to reject glaring outliers and replace them with previous fits. This should offer a smoother output.
