import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load images
image1 = cv2.imread('./data1/obj1_5.JPG')
image2 = cv2.imread('./data1/obj1_t1.JPG')

surf = cv2.xfeatures2d.SURF_create(5000)

keypoints1, descriptors1 = surf.detectAndCompute(image1, None)
keypoints2, descriptors2 = surf.detectAndCompute(image2, None)
print(len(keypoints1))

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

def match_features_with_ratio(descriptors1, descriptors2, ratio_threshold):
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    return good_matches

def draw_matches_knn(img1, keypoints1, img2, keypoints2, matches):

    matches_knn = [[m] for m in matches]
    img1_with_keypoints = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_with_keypoints = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    matched_img = cv2.drawMatchesKnn(img1_with_keypoints, keypoints1, img2_with_keypoints, keypoints2, matches_knn, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_img

optimal_ratio_threshold = 0.75
#suboptimal_ratio_threshold = 0.9

optimal_matches = match_features_with_ratio(descriptors1, descriptors2, optimal_ratio_threshold)
print(len(optimal_matches))
#suboptimal_matches = match_features_with_ratio(descriptors1, descriptors2, suboptimal_ratio_threshold)

optimal_matched_img = draw_matches_knn(image1, keypoints1, image2, keypoints2, optimal_matches)
#suboptimal_matched_img = draw_matches_knn(image1, keypoints1, image2, keypoints2, suboptimal_matches)


plt.figure(figsize=(16, 8))


plt.title(f'Optimal Result (Ratio Threshold = {optimal_ratio_threshold})')
plt.imshow(cv2.cvtColor(optimal_matched_img, cv2.COLOR_BGR2RGB))
plt.axis('off')


plt.show()
