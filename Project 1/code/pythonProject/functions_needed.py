import cv2
import numpy as np
import matplotlib.pyplot as plt

# 旋转图像
def rotate_image(image, angle):
    (h, w) = image.shape[:2] #get length and width
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0) #rotation matrix
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# 计算关键点匹配的重复率
def compute_repeatability(detector, img1, img2, angle):
    # Detect keypoints and descriptors in original and rotated images
    kp1, des1 = detector.detectAndCompute(img1, None) #original image
    kp2, des2 = detector.detectAndCompute(img2, None) #rotated image
    # Predict the position of each key point in the rotated image
    predicted_kp1 = []
    (h, w) = img1.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    for kp in kp1:
        pt = np.array([kp.pt[0], kp.pt[1], 1.0]) #to array
        predicted_pt = np.dot(M, pt) #where it should be after rotation
        predicted_kp1.append((predicted_pt[0], predicted_pt[1]))
    # Check if the keypoint is in the neighborhood of the predicted position
    repeatable_count = 0
    for (x1, y1) in predicted_kp1:
        for kp in kp2:
            x2, y2 = kp.pt
            if abs(x2 - x1) <= 2 and abs(y2 - y1) <= 2:
                repeatable_count += 1
                break
    # compute repeatability
    repeatability = repeatable_count / len(kp1) # if len(kp1) > 0 else 0
    return repeatability

# 绘制重复率曲线
def plot_repeatability(rotation_angles, repeatability):
    plt.plot(rotation_angles, repeatability)
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Repeatability')
    plt.title('Repeatability vs Rotation Angle')
    plt.grid(True)
    plt.show()

def plot_repeatability_scale(scale_factors, repeatability):
    plt.plot(scale_factors, repeatability)
    plt.xlabel('Scaling Factor')
    plt.ylabel('Repeatability')
    plt.title('Repeatability vs Scaling Factor')
    plt.grid(True)
    plt.show()

def compute_repeatability_scale(detector, img1, img2):
    # Detect keypoints and descriptors in original and scaled images
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    # Mapping to the coordinates of the scaled image
    scale = img2.shape[0] / img1.shape[0]
    predicted_kp1 = [(kp.pt[0] * scale, kp.pt[1] * scale) for kp in kp1]
    # compute repeatability
    repeatable_count = 0
    for (x1, y1) in predicted_kp1:
        found_match = False
        for kp in kp2:
            x2, y2 = kp.pt
            if abs(x2 - x1) <= 2 and abs(y2 - y1) <= 2:
                found_match = True
                break
        if found_match:
            repeatable_count += 1
    repeatability = repeatable_count / len(kp1)
    return repeatability
