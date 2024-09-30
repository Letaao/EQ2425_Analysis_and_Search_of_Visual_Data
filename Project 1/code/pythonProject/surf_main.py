import cv2
import numpy as np
import functions_needed

image = cv2.imread('./data1/obj1_5.JPG', cv2.IMREAD_GRAYSCALE)

# create surf detector
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=5000)
# detect keypoints and descriptors
keypoints, descriptors = surf.detectAndCompute(image, None)
print("The number of keypoints:", len(keypoints))
# draw
g = (0, 255, 0)  # green BGR, not RGB
img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=g)
# save file
cv2.imwrite('surf_keypoints.jpg', img_with_keypoints)

rotation_angles = np.arange(0, 361, 15) #include 360
surf_repeatability = []

# 计算每个角度的重复率
for angle in rotation_angles:
    rotated_image = functions_needed.rotate_image(image, angle)
    surf_rep = functions_needed.compute_repeatability(surf, image, rotated_image, angle)
    surf_repeatability.append(surf_rep)

# 绘制结果
functions_needed.plot_repeatability(rotation_angles, surf_repeatability)

#2.2.3
surf_repeatability_scale = []
scale_factors = [1.2 ** i for i in range(9)]
for scale in scale_factors:
    # 缩放图像
    scaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    repeatability_surf_scale = functions_needed.compute_repeatability_scale(surf, image, scaled_image)
    surf_repeatability_scale.append(repeatability_surf_scale)

functions_needed.plot_repeatability_scale(scale_factors, surf_repeatability_scale)