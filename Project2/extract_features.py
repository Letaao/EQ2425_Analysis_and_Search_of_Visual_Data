import cv2
import os
import numpy as np
import pickle
# load images
server_folder = 'Data2/server/'
client_folder = 'Data2/client/'

sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)

def extract_sift_features(image_path):

    img = cv2.imread(image_path)

    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

object_features_server = {}
object_features_client = {}

# server
for filename in os.listdir(server_folder):

    if filename.endswith('.JPG'):
        object_number = filename.split('obj')[1].split('_')[0]

        image_path = os.path.join(server_folder, filename)
        keypoints, descriptors = extract_sift_features(image_path)
        #print(keypoints)


        if descriptors is not None:

            if object_number not in object_features_server:
                object_features_server[object_number] = descriptors
            else:
                object_features_server[object_number] = np.vstack((object_features_server[object_number], descriptors))

# Report
total_features = 0
for obj, features in object_features_server.items():
    num_features = features.shape[0]
    print(f"Server object {obj} has {num_features} features.")
    total_features += num_features

average_features = total_features / len(object_features_server)
print(f"Average number of SIFT features per server object: {average_features:.2f}")


#client

for filename in os.listdir(client_folder):

    if filename.endswith('.JPG'):

        object_number = filename.split('obj')[1].split('_')[0]

        image_path = os.path.join(client_folder, filename)
        keypoints, descriptors = extract_sift_features(image_path)



        if descriptors is not None:

            if object_number not in object_features_client:
                object_features_client[object_number] = descriptors
            else:
                object_features_client[object_number] = np.vstack((object_features_client[object_number], descriptors))

# Report
print(object_features_client)
total_features_client = 0
for obj, features in object_features_client.items():
    num_features = features.shape[0]
    print(f"Client object {obj} has {num_features} features.")
    total_features_client += num_features

average_features = total_features_client / len(object_features_client)
print(f"Average number of SIFT features per client object: {average_features:.2f}")


with open('object_features_client.pkl', 'wb') as f_client:
    pickle.dump(object_features_client, f_client)

with open('object_features_server.pkl', 'wb') as f_server:
    pickle.dump(object_features_server, f_server)

print("Features saved successfully.")