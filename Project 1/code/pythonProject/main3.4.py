import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images

image1 = cv2.imread('./data1/obj1_5.JPG')
image2 = cv2.imread('./data1/obj1_t1.JPG')

sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)

keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
print(len(keypoints1))
print(len(keypoints2))

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

def match_features_with_ratio(descriptors1, descriptors2, ratio_threshold):
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    return good_matches

def draw_matches_with_lines(img1, keypoints1, img2, keypoints2, matches):

    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    combined_img = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)
    combined_img[:height1, :width1] = img1
    combined_img[:height2, width1:] = img2

    for match in matches:
        pt1 = (int(keypoints1[match.queryIdx].pt[0]), int(keypoints1[match.queryIdx].pt[1]))
        pt2 = (int(keypoints2[match.trainIdx].pt[0]) + width1, int(keypoints2[match.trainIdx].pt[1]))

        cv2.circle(combined_img, pt1, 5, (0, 255, 0), -1)
        cv2.circle(combined_img, pt2, 5, (0, 255, 0), -1)

        cv2.line(combined_img, pt1, pt2, (255, 0, 0), 1)

    return combined_img

optimal_ratio_threshold = 0.75
suboptimal_ratio_threshold = 0.9

optimal_matches = match_features_with_ratio(descriptors1, descriptors2, optimal_ratio_threshold)
print(len(optimal_matches))
suboptimal_matches = match_features_with_ratio(descriptors1, descriptors2, suboptimal_ratio_threshold)
print(len(suboptimal_matches))

optimal_matched_img = draw_matches_with_lines(image1, keypoints1, image2, keypoints2, optimal_matches)
suboptimal_matched_img = draw_matches_with_lines(image1, keypoints1, image2, keypoints2, suboptimal_matches)

plt.figure(figsize=(16, 8))

# Optimal result
plt.subplot(1, 2, 1)
plt.title(f'Optimal Result (Ratio Threshold = {optimal_ratio_threshold})')
plt.imshow(cv2.cvtColor(optimal_matched_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Suboptimal result
plt.subplot(1, 2, 2)
plt.title(f'Suboptimal Result (Ratio Threshold = {suboptimal_ratio_threshold})')
plt.imshow(cv2.cvtColor(suboptimal_matched_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
