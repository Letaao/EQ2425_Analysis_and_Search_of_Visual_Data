import cv2
import numpy as np
import functions_needed

# 加载图像
image = cv2.imread('./data1/obj1_5.JPG', cv2.IMREAD_GRAYSCALE)

# create sift detector
sift = cv2.SIFT_create(contrastThreshold=0.17, edgeThreshold=10)
# detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(image, None)
print("The number of keypoints:", len(keypoints))
# draw
g = (0, 255, 0)  # green BGR, not RGB
img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=g)
# save file
cv2.imwrite('sift_keypoints.jpg', img_with_keypoints)

rotation_angles = np.arange(0, 361, 15) #include 360
sift_repeatability = []

# 计算每个角度的重复率
for angle in rotation_angles:
    rotated_image = functions_needed.rotate_image(image, angle)
    sift_rep = functions_needed.compute_repeatability(sift, image, rotated_image, angle)
    sift_repeatability.append(sift_rep)

# 绘制结果
functions_needed.plot_repeatability(rotation_angles, sift_repeatability)


#2.2.3
sift_repeatability_scale = []
scale_factors = [1.2 ** i for i in range(9)]
for scale in scale_factors:
    # 缩放图像
    scaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    repeatability_sift_scale = functions_needed.compute_repeatability_scale(sift, image, scaled_image)
    sift_repeatability_scale.append(repeatability_sift_scale)

functions_needed.plot_repeatability_scale(scale_factors, sift_repeatability_scale)
