import cv2
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('./data1/obj1_5.JPG')
img2 = cv2.imread('./data1/obj1_t1.JPG')

# Create a SIFT detector with custom parameters
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)

# Detect SIFT keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Create a BFMatcher object with default params
bf = cv2.BFMatcher(cv2.NORM_L2)

# Function to apply fixed threshold matching
def match_keypoints(descriptors1, descriptors2, threshold):
    matches = bf.match(descriptors1, descriptors2)
    # Filter matches based on distance threshold
    good_matches = [m for m in matches if m.distance < threshold]
    return good_matches

# Function to draw matches
def draw_matches(img1, img2, keypoints1, keypoints2, matches, title, save_path=None):
    img_matches = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, matches, None,
        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

# Define distance thresholds
thresholds = [75, 70]  # Example thresholds, adjust as needed

# Match keypoints and draw results
matches_best = match_keypoints(descriptors1, descriptors2, thresholds[0])
matches_suboptimal = match_keypoints(descriptors1, descriptors2, thresholds[1])

# Plot the results
draw_matches(img1, img2, keypoints1, keypoints2, matches_best, None, 'Optimal Result with Threshold = {}'.format(thresholds[0]))
draw_matches(img1, img2, keypoints1, keypoints2, matches_suboptimal, None, 'Suboptimal Result with Threshold = {}'.format(thresholds[1]))
