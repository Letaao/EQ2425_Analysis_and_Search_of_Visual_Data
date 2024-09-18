import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('./data1/obj1_5.JPG')
img2 = cv2.imread('./data1/obj1_t1.JPG')

# Initialize
sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, nfeatures=1000)

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)


# Function to draw matches
def draw_matches(img1, kp1, img2, kp2, matches):
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        matchesMask=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig('Nearest Neighbor Matching')
    plt.show()


draw_matches(img1, kp1, img2, kp2, matches)
