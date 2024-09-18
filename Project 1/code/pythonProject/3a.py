import cv2
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('./data1/obj1_5.JPG')
img2 = cv2.imread('./data1/obj1_t1.JPG')

# Create a SIFT detector with custom parameters
sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, nfeatures=500)

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Convert images to RGB for matplotlib
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Plot keypoints
def plot_keypoints(image_rgb, keypoints, title, edge_color='green', face_color='none', size=100, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    x_coords = [kp.pt[0] for kp in keypoints]
    y_coords = [kp.pt[1] for kp in keypoints]
    plt.scatter(x_coords, y_coords, edgecolor=edge_color, facecolor=face_color, s=size, linewidth=0.8)
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

# Plot keypoints
plot_keypoints(img1_rgb, keypoints1, None, edge_color='green', face_color='none', size=50, save_path='SIFT Features on obj1_5.JPG')
plot_keypoints(img2_rgb, keypoints2, None, edge_color='green', face_color='none', size=50, save_path='SIFT Features on obj1_t1.JPG')

