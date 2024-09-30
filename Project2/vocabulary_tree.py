import numpy as np
from sklearn.cluster import KMeans
import pickle
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class TreeNode:
    def __init__(self, cluster_center=None):
        self.cluster_center = cluster_center
        self.children = []
        self.is_leaf = False


def hi_kmeans(data, b, depth, current_depth=0):

    node = TreeNode()

    if current_depth == depth or len(data) <= b:
        node.is_leaf = True
        return node


    kmeans = KMeans(n_clusters=b, random_state=0).fit(data)

    node.cluster_center = kmeans.cluster_centers_

    for i in range(b):

        cluster_data = data[kmeans.labels_ == i]

        child_node = hi_kmeans(cluster_data, b, depth, current_depth + 1)

        node.children.append(child_node)

    return node

with open('object_features_client.pkl', 'rb') as f_client:
    object_features_client = pickle.load(f_client)

with open('object_features_server.pkl', 'rb') as f_server:
    object_features_server = pickle.load(f_server)


object_features_server_list = list(object_features_server.values())

object_features_client_list = list(object_features_client.values())


def traverse_tree(tree, descriptor):

    current_node = tree
    while not current_node.is_leaf:
        closest_idx = np.argmin([np.linalg.norm(descriptor - center) for center in current_node.cluster_center])
        current_node = current_node.children[closest_idx]
    return current_node


def tf_idf_ranking(tree, query_descriptors, database_descriptors, num_objects, num_clusters):

    query_leaves = defaultdict(int)
    for descriptor in query_descriptors:
        leaf_node = traverse_tree(tree, descriptor)
        query_leaves[leaf_node] += 1

    database_leaves = [defaultdict(int) for _ in range(num_objects)]
    for i in range(num_objects):
        for descriptor in database_descriptors[i]:
            leaf_node = traverse_tree(tree, descriptor)
            database_leaves[i][leaf_node] += 1

    idf = np.log(
        num_objects / (1 + np.array([sum(1 for obj in database_leaves if leaf in obj) for leaf in query_leaves])))

    scores = []
    for i in range(num_objects):
        tf_query = np.array(list(query_leaves.values()))
        tf_db = np.array([database_leaves[i].get(leaf, 0) for leaf in query_leaves])
        tf_idf_score = np.sum(tf_query * tf_db * idf)
        scores.append((i, tf_idf_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scores]

#Bonus: smoothed TF-IDF Formula
'''
def tf_idf_ranking(tree, query_descriptors, database_descriptors, num_objects,num_clusters):

    query_leaves = defaultdict(int)
    for descriptor in query_descriptors:
        leaf_node = traverse_tree(tree, descriptor)
        query_leaves[leaf_node] += 1

    database_leaves = [defaultdict(int) for _ in range(num_objects)]
    for i in range(num_objects):
        for descriptor in database_descriptors[i]:
            leaf_node = traverse_tree(tree, descriptor)
            database_leaves[i][leaf_node] += 1

    idf = np.log(
        num_objects / (1 + np.array([sum(1 for obj in database_leaves if leaf in obj) for leaf in query_leaves])))

    scores = []
    for i in range(num_objects):

        tf_query = 1 + np.log(np.array(list(query_leaves.values())) + 1e-10)  # 添加小值避免0

        tf_db = 1 + np.log(np.array([database_leaves[i].get(leaf, 0) for leaf in query_leaves]) + 1e-10)  # 添加小值避免0

        tf_idf_score = np.sum(tf_query * tf_db * idf)
        scores.append((i, tf_idf_score))


    scores.sort(key=lambda x: x[1], reverse=True)


    return [x[0] for x in scores]
'''

def calculate_recall(query_descriptors, database_descriptors, correct_object_idx, tree, top_k):

    ranked_objects = tf_idf_ranking(tree, query_descriptors, database_descriptors, len(database_descriptors), top_k)
    return 1 if correct_object_idx in ranked_objects[:top_k] else 0


def test_vocabulary_tree(tree, queries, database, correct_indices, top_k=5):

    top1_recall = 0
    top5_recall = 0
    num_queries = len(queries)

    for i in range(num_queries):
        query_descriptors = queries[i]
        correct_object_idx = correct_indices[i]

        top1_recall += calculate_recall(query_descriptors, database, correct_object_idx, tree, top_k=1)

        top5_recall += calculate_recall(query_descriptors, database, correct_object_idx, tree, top_k=5)

    return top1_recall / num_queries, top5_recall / num_queries


#4.(a)

trees_config = [(4, 3), (4, 5), (5, 7)]

for b, depth in trees_config:

    tree = hi_kmeans(np.vstack(object_features_server_list ), b, depth)

    correct_indices = [i for i in range(50)]
    top1, top5 = test_vocabulary_tree(tree, object_features_client_list, object_features_server_list , correct_indices)

    print(f"For b = {b}, depth = {depth}:")
    print(f"Top-1 Recall: {top1:.2f}")
    print(f"Top-5 Recall: {top5:.2f}")

#4.(b)
def test_vocabulary_tree_percentage(tree, queries, database, correct_indices, percentages):

    results = {percent: {'top1': 0, 'top5': 0} for percent in percentages}
    num_queries = len(queries)

    for i in range(num_queries):
        query_descriptors = queries[i]
        correct_object_idx = correct_indices[i]

        for percent in percentages:
            num_features = int(len(query_descriptors) * (percent / 100))
            selected_descriptors = query_descriptors[:num_features]

            results[percent]['top1'] += calculate_recall(selected_descriptors, database, correct_object_idx, tree, top_k=1)
            results[percent]['top5'] += calculate_recall(selected_descriptors, database, correct_object_idx, tree, top_k=5)

    for percent in percentages:
        results[percent]['top1'] /= num_queries
        results[percent]['top5'] /= num_queries

    return results


b = 5
depth = 7

tree = hi_kmeans(np.vstack(object_features_server_list), b, depth)

percentages = [90, 70, 50]
correct_indices = [i for i in range(50)]
results = test_vocabulary_tree_percentage(tree, object_features_client_list, object_features_server_list, correct_indices, percentages)

for percent in percentages:
    print(f"For {percent}% of features:")
    print(f"  Top-1 Recall: {results[percent]['top1']:.2f}")
    print(f"  Top-5 Recall: {results[percent]['top5']:.2f}")


#Bonus

tree = hi_kmeans(np.vstack(object_features_server_list), 5, 7)

correct_indices = [i for i in range(50)]
top1, top5 = test_vocabulary_tree(tree, object_features_client_list, object_features_server_list, correct_indices)

print(f"For b = {5}, depth = {7}:")
print(f"Top-1 Recall: {top1:.2f}")
print(f"Top-5 Recall: {top5:.2f}")
