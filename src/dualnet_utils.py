import numpy as np
from scipy.sparse.csgraph import dijkstra


# Super-adjacency matrix calculation
def st_mul_layer(g_1, g_2, g_1_x, g_2_x, top_k, g_1_x_pad):
    """
    Computes the super-adjacency matrix for two graphs.

    This function calculates the Euclidean distances between nodes of two graphs, applies a softmax function to convert
    these distances into weights, and then constructs a super-adjacency matrix that combines the structure information
    of both graphs by connecting the top_k closest node pairs.

    Parameters:
    g_1: Adjacency matrix of the first graph, shape (n_nodes_1, n_nodes_1).
    g_2: Adjacency matrix of the second graph, shape (n_nodes_2, n_nodes_2).
    top_k: Number of top closest node pairs to select for forming edges in the super-adjacency matrix.

    Returns:
    unit_matrix: The super-adjacency matrix combining the structure information of g_1 and g_2.
    """

    # Calculate Euclidean distances between nodes of g_1 and g_2
    distances = np.linalg.norm(g_1_x[:, np.newaxis] - g_2_x, axis=2)

    # Apply softmax function to convert distances into weights
    exp_distances = np.exp(-distances)
    softmax_distances = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)

    # Sort the weights to select the top_k largest ones
    softmax_sorted_distances_indices = np.argsort(-softmax_distances, axis=None)

    # Select indices of the top_k largest weights
    top_10_indices = softmax_sorted_distances_indices[:top_k]
    top_10_row_indices, top_10_col_indices = np.unravel_index(top_10_indices, softmax_distances.shape)


    # Create a zero matrix with size equal to the sum of nodes in both graphs
    unit_matrix_size = g_1.shape[0] + g_2.shape[0]
    unit_matrix = np.zeros((unit_matrix_size, unit_matrix_size)) + np.eye(g_1.shape[0] + g_2.shape[0])

    # Place the original adjacency matrices into the unit_matrix
    unit_matrix[:g_1.shape[0], :g_1.shape[0]] = g_1
    unit_matrix[g_1.shape[0]:, g_1.shape[0]:] = g_2

    # Update the unit_matrix based on the selected top_k weights to form the super-adjacency matrix
    inter_layer = []
    for row_idx, col_idx in zip(top_10_row_indices, top_10_col_indices):
        adjusted_col_idx = col_idx + g_1.shape[0]
        adjusted_row_idx = row_idx
        inter_layer.append((adjusted_row_idx,adjusted_col_idx))
        unit_matrix[adjusted_row_idx, adjusted_col_idx] = softmax_distances[row_idx, col_idx]
        unit_matrix[adjusted_col_idx, adjusted_row_idx] = softmax_distances[row_idx, col_idx]
    # unit_matrix_ori = unit_matrix
    # unit_matrix_ori[unit_matrix_ori == 0.5000] = 0
    # unit_matrix_ori[unit_matrix_ori > 0] = 1
    # print(unit_matrix)
    return unit_matrix, inter_layer,unit_matrix[:g_1.shape[0], :g_1.shape[0]], unit_matrix[g_1.shape[0]:, g_1.shape[0]:]


def layer_degree_correlation(adjacency_matrices):
    """
    The correlation coefficient between layers of multi-layer network is calculated.

    Parameters:
    adjacency_matrices: A list where each element is a one-layer adjacency matrix.

    return:
    layer_correlations: A matrix where the elements (i, j) represent the degree correlation coefficient between layers i and j.
    """
    num_layers = len(adjacency_matrices)
    layer_correlations = np.zeros((num_layers, num_layers))

    # 计算每一层的度
    degrees = [np.sum(adj, axis=1) for adj in adjacency_matrices]

    for i in range(num_layers):
        for j in range(i, num_layers):
            # 计算层i和层j之间的度相关系数

            correlation = np.corrcoef(degrees[i], degrees[j])[0, 1]
            layer_correlations[i, j] = correlation
            layer_correlations[j, i] = correlation  # 度相关系数矩阵是对称的
    # print(correlation)
    return layer_correlations[0, 1]


def interlayer_connection_density(super_adjacency_matrix, num_nodes_per_layer):

    num_layers = len(num_nodes_per_layer)

    densities = []
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            # 计算层i和层j之间的连接矩阵
            start_row = sum(num_nodes_per_layer[:i])
            end_row = start_row + num_nodes_per_layer[i]
            start_col = sum(num_nodes_per_layer[:j])
            end_col = start_col + num_nodes_per_layer[j]


            interlayer_adj = super_adjacency_matrix[start_row:end_row, start_col:end_col]

            num_interlayer_edges = np.sum(interlayer_adj)


            max_possible_edges = num_nodes_per_layer[i] * num_nodes_per_layer[j]


            density = num_interlayer_edges / max_possible_edges
            densities.append((i, j, density))

    return densities[0][-1]


def interlayer_clustering_coefficient(super_adjacency_matrix, num_nodes_per_layer):

    num_layers = len(num_nodes_per_layer)
    clustering_coeffs = {}
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            # 计算层i和层j之间的连接矩阵
            start_row_i = sum(num_nodes_per_layer[:i])
            end_row_i = start_row_i + num_nodes_per_layer[i]
            start_row_j = sum(num_nodes_per_layer[:j])
            end_row_j = start_row_j + num_nodes_per_layer[j]


            interlayer_adj = super_adjacency_matrix[start_row_i:end_row_i, start_row_j:end_row_j]


            num_interlayer_edges = np.sum(interlayer_adj)

            if num_interlayer_edges == 0:

                clustering_coeffs[(i, j)] = 0.0
                continue


            triangles = 0
            for k in range(num_nodes_per_layer[i]):
                for l in range(num_nodes_per_layer[j]):
                    if interlayer_adj[k, l] == 1:
                        neighbors_i = np.sum(super_adjacency_matrix[start_row_i:end_row_i, start_row_i:end_row_i][k, :])
                        neighbors_j = np.sum(super_adjacency_matrix[start_row_j:end_row_j, start_row_j:end_row_j][l, :])
                        triangles += interlayer_adj[k, :].dot(interlayer_adj[:, l])

            clustering_coeff = (2 * triangles) / (num_interlayer_edges * (num_interlayer_edges - 1))
            clustering_coeffs[(i, j)] = clustering_coeff

    return clustering_coeffs[list(clustering_coeffs.keys())[0]]


def calculate_average_interlayer_path_length(super_adjacency_matrix, num_nodes_per_layer):
    num_layers = len(num_nodes_per_layer)
    total_path_length = 0
    total_paths = 0

    for layer in range(num_layers):
        for node in range(num_nodes_per_layer[layer]):
            paths = dijkstra(super_adjacency_matrix, directed=False, indices=layer * num_nodes_per_layer[layer] + node)
            paths[np.isinf(paths)] = 0
            for target_layer in range(num_layers):
                if target_layer != layer:
                    for target_node in range(num_nodes_per_layer[target_layer]):
                        target_index = target_layer * num_nodes_per_layer[target_layer] + target_node
                        total_path_length += paths[target_index]
                        total_paths += 1

    average_interlayer_path_length = total_path_length / total_paths if total_paths > 0 else 0
    return average_interlayer_path_length


def calculate_interlayer_alignment(superadjacency_matrix, ideal_alignment_matrix):

    dot_product = np.sum(superadjacency_matrix * ideal_alignment_matrix)

    norm_superadjacency = np.linalg.norm(superadjacency_matrix)
    norm_ideal = np.linalg.norm(ideal_alignment_matrix)
    alignment_score = dot_product / (norm_superadjacency * norm_ideal)

    return alignment_score
