from .utils import Atom, Residue, ActiveSite

# JAMES
import prody as pd
import numpy as np
import scipy as sp
from scipy.cluster import hierarchy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import collections
from .utils import ClusterPartition
from.io import write_hierarchical_clustering
import pprint

def compute_similarity(site_a, site_b):
    """
    Computes the similarity between two given ActiveSite instances.
    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)

    Similarity Metric: Here I am computing the distance between two numpy matricies containing counts for atom types
    C, O, N, and S.

    After some discussion with a lab mate, I decided to use a single metric instead of needing to arbitrarily
    weight and combine several unrelated metrics (e.g. residue types/counts, atom types/counts, active site volume).
    Both the composition of residues in enzyme active sites and their positioning relative to each other are extremely
    important for determining function. Therefore, I wanted my metric to combine both structural information with the
    chemical composition of the active site.

    To do this, I construct an Nx4 matrix for each active site. Starting from the centroid of the active site, I take
    steps of length L and record the number of each atom type in the sphere of radius L. I will make counts in shells,
    meaning I will only count new atoms encountered with each increment in L.

    Atoms  | C | O | N | S |
    ------------------------
    Step 1 | X | X | X | X |
    ------------------------
    Step 2 | X | X | X | X |
    ------------------------
    Step 3 |  ...and so on

    Example:

    Active Site Atom counts matrix:
    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  1.  0.  0.]
     [ 2.  1.  1.  0.]
     [ 0.  0.  0.  0.]
     [ 2.  0.  1.  0.]
     [ 2.  2.  0.  0.]
     [ 2.  1.  1.  0.]
     [ 2.  0.  0.  0.]
     [ 2.  1.  0.  0.]
     [ 1.  2.  2.  0.]
     [ 3.  0.  2.  0.]
     [ 1.  1.  0.  0.]
     [ 2.  0.  0.  0.]
     [ 3.  0.  1.  0.]
     [ 2.  1.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 4.  2.  1.  0.]]

    Mean cluster matrix:
    [[ 0.          0.          0.          0.        ]
     [ 0.09090909  0.          0.01818182  0.        ]
     [ 0.27272727  0.12727273  0.4         0.        ]
     [ 0.23636364  0.18181818  0.30909091  0.        ]
     [ 0.81818182  0.30909091  0.25454545  0.        ]
     [ 1.23636364  0.85454545  0.67272727  0.01818182]
     [ 1.25454545  0.56363636  0.72727273  0.        ]
     [ 1.38181818  0.32727273  0.72727273  0.        ]
     [ 2.43636364  0.50909091  0.81818182  0.        ]
     [ 3.38181818  0.47272727  0.52727273  0.        ]
     [ 3.16363636  1.05454545  1.10909091  0.        ]
     [ 3.69090909  0.94545455  1.58181818  0.        ]
     [ 3.10909091  0.69090909  0.98181818  0.        ]
     [ 2.2         0.70909091  0.98181818  0.        ]
     [ 1.94545455  0.41818182  1.45454545  0.        ]
     [ 1.34545455  0.54545455  0.36363636  0.        ]
     [ 1.23636364  0.43636364  0.23636364  0.        ]
     [ 0.61818182  0.56363636  0.18181818  0.        ]
     [ 0.8         0.67272727  0.2         0.        ]
     [ 0.10909091  0.52727273  0.09090909  0.        ]
     [ 0.16363636  0.27272727  0.01818182  0.        ]]

    Distance:
    7.50166923573
    """

    similarity = np.linalg.norm(site_a - site_b)
    return similarity


def cluster_by_partitioning(active_sites, cluster_number):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances, number of desired clusters
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)

    1. Pick a number of cluster centers randomly
    2. Iterate through active sites and assign each active site to a cluster
    3. Recalculate cluster centers after all additions
    4. Back to 2. until termination criteria are met (clusters do not change from previous iteration)
    """

    ########################################
    # Implement Clustering by Partitioning #
    ########################################

    cluster_centers = []

    # Select random active site coordinates to serve as initial cluster centers

    while len(cluster_centers) != cluster_number:
        active_site_pick = active_sites[np.random.randint(0, len(active_sites))]
        if active_site_pick not in cluster_centers:
            cluster_centers.append(active_site_pick)

    # Instantiate cluster centers with coordinates and add to current_clusters
    current_clusters = []
    ID = 0
    for cluster_center in cluster_centers:
        K_mean = ClusterPartition()
        K_mean.cluster_ID = "Cluster-{}".format(ID)
        ID += 1
        K_mean.cluster_center = cluster_center.shell_matrix
        current_clusters.append(K_mean)

    # Actually start clustering things
    stop = False
    iter_count = 0

    while stop == False:
        iter_count += 1
        print("Current iteration: {}".format(iter_count))

        # Clear cluster_members_current lists from cluster objects
        for center in current_clusters:
            center.cluster_members_current = []

        # Iterate through active sites and assign to clusters
        for active_site in active_sites:
            calculated_distances = {center.cluster_ID: compute_similarity(active_site.shell_matrix, center.cluster_center) for center in current_clusters}

            min_distance_cluster = min(calculated_distances, key=(lambda key: calculated_distances[key]))
            # print("Min distance: {} - {}".format(min_distance_cluster, calculated_distances[min_distance_cluster]))

            for center in current_clusters:
                if center.cluster_ID == min_distance_cluster:
                    center.cluster_members_current.append(active_site)
                    break

        # Update cluster centers
        stop_counter = 0

        for center in current_clusters:
            center.cluster_center = np.mean(np.array([active_site.shell_matrix for active_site in center.cluster_members_current]), axis=0)

            # Update Break condition
            if center.cluster_members_previous == center.cluster_members_current:
                stop_counter += 1
                # print("Stop counter incremented!")

            center.cluster_members_previous = center.cluster_members_current

        # Break Condition
        if stop_counter == len(current_clusters):
            stop = True

    print("Termination criteria reached after {} iterations".format(iter_count))

    return_clusters = []

    for cluster in current_clusters:
        current_cluster = [site.name for site in cluster.cluster_members_current]
        return_clusters.append(current_cluster)

    return return_clusters


def cluster_hierarchically(active_sites, number_of_clusters):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.

    I need a bunch of different matricies to record data for this clustering:
    1. A distance matrix (implemented as pandas dataframe... clunky but it will work)
    2. A 4 x N matrix to record which clusters I join together, their distance, and how many members are in the cluster
    3. A dict that records cluster members in lists

    I need to update the distance matrix every time I create a new cluster. One of the reasons I opted to go with the
    pandas dataframe in lieu of a numpy array is that I can use index labels instead of needing to keep track of indices
    every time I update the matrix where things will move around.

    I am recording cluster joinings in a 4 x N matrix so that I can use the scipy dendrogram at the end of this to
    evaluate my clusterings

    I am using centroid linkaging to evaluate my cluster distances as it is intermediate between single linkage (which
    produces drawn out clusters) and complete linkage (which forms compact cluster). Centroid linkaging is also
    relatively easy to implement, especially since I do something similar for my K-means clustering where I calcualte
    cluster centroids.

    Input:
    a list of ActiveSite instances
    number_of_clusters: Number of desired clusters as inferred from inspection of dendrogram and elbow chart

    Output: a list of clusterings
    """
    ########################
    # Initialize variables #
    ########################

    dendrogram_list = []
    cluster_index = 0
    cluster_dict = {}
    clusters_record = {}
    cluster_indicies_dict = collections.OrderedDict()

    # Keep track of cluster members in dict of lists
    # Create dict linking numerical indices to cluster names
    # Keep a record of all clusters created to retrieve at the end
    for active_site in active_sites:
        cluster_dict[active_site.name] = [active_site]
        clusters_record[cluster_index] = [active_site]
        cluster_indicies_dict[active_site.name] = cluster_index
        cluster_index += 1

    # Calculate all pairwise distances between active sites
    # Distance matrix represented in a Pandas Dataframe
    df = _calculate_pairwise_distances(active_sites)

    # Output for debugging
    df.to_csv("_calculate_pairwise_distances.csv")

    ###################################
    # Perform Hierarchical Clustering #
    ###################################

    # Terminate when cluster_dict contains only one entry

    while len(cluster_dict) > 1:

        # Find the minimum distance between two clusters (initially singletons) currently in play
        # row_values = df.min(axis=0) # Row-wise, outputs column index of the cluster with the minimum distance from the row index
        row_min_indicies = df.min(axis=0) # Row-wise, outputs minimum distance from the row index PDB
        min_cluster_a = row_min_indicies.idxmin(axis=0) # Outputs index of cluster with minimum distance in
        min_cluster_b_series = df.idxmin(axis=0) # Row-wise, outputs column index of the cluster with the minimum distance from the row index

        first_cluster = min_cluster_a
        second_cluster = min_cluster_b_series[first_cluster]

        # print(first_cluster) # ID of first cluster
        # print(second_cluster) # ID of scecond cluster
        # print(df.ix[second_cluster, first_cluster]) # [row, column], distance

        # Combine the two clusters
        # Record new cluster members in cluster_dict
        new_cluster_index = cluster_index
        cluster_dict[new_cluster_index] = list(cluster_dict[first_cluster] + cluster_dict[second_cluster])
        clusters_record[new_cluster_index] = cluster_dict[new_cluster_index]

        # Record new cluster name to cluster index
        cluster_indicies_dict[new_cluster_index] = cluster_index
        # cluster_index += 1

        # Remove old clusters from cluster dict
        cluster_dict.pop(first_cluster)
        cluster_dict.pop(second_cluster)

        # Record new cluster in 4 x N matrix
        # [Cluster 1, cluster 2, cluster centroid distance, number of members in new cluster]
        dendrogram_list.append((cluster_indicies_dict[first_cluster],
                                cluster_indicies_dict[second_cluster],
                                df.ix[second_cluster, first_cluster],
                                len(cluster_dict[new_cluster_index])
                                )
                               )

        # Calculate centroid of new cluster,
        new_cluster_centroid = np.mean(np.array([active_site.shell_matrix for active_site in cluster_dict[new_cluster_index]]), axis=0)

        # Delete old clusters from distance matrix
        # Add new cluster to distance matrix and calcuate new distances
        df = df.drop([first_cluster, second_cluster])
        df = df.drop([first_cluster, second_cluster], axis=1)

        new_distances_list = []
        distance_series_index = []

        for index, values in df.iterrows():
            current_cluster_centroid = np.mean(np.array([active_site.shell_matrix for active_site in cluster_dict[index]]), axis=0)
            new_distances_list.append(compute_similarity(new_cluster_centroid, current_cluster_centroid))
            distance_series_index.append(index)

        new_cluster_series_column = pd.Series(new_distances_list, name=new_cluster_index, index=distance_series_index)
        new_cluster_series_row = pd.Series(name=new_cluster_index, index=distance_series_index)

        df = pd.concat([df, new_cluster_series_column], axis=1)
        df = df.append(new_cluster_series_row)

        cluster_index += 1

    _plot_elbow_chart(dendrogram_list)
    _plot_dendrogram(dendrogram_list, active_sites, cluster_indicies_dict)

    # Return clusters based on user input
    # User input *should* be informed by inspection of dendrogram and elbow chart
    # clusters = 4 for the active site dataset, as jumps increase from an average of ~10 to 15.43 -> 16.77 -> 35.23
    cluster_set = set()
    for join in dendrogram_list[ -(number_of_clusters - 1):]:
        cluster_set.add(join[0])
        cluster_set.add(join[1])

    # N number of initial clusters
    # N - 1 of clusters made during clustering
    number_of_total_clusters = len(active_sites) * 2 - 1
    clusters_to_exclude = set(range(number_of_total_clusters - number_of_clusters + 1, number_of_total_clusters - 1, 1))
    clusters_to_return = cluster_set ^ clusters_to_exclude

    return_list_of_clusters = []
    for return_cluster in clusters_to_return:
        cluster_members = [cluster_member.name for cluster_member in clusters_record[return_cluster]]
        return_list_of_clusters.append(cluster_members)

    write_hierarchical_clustering("All_hierarchical_clusters.csv", clusters_record)

    return return_list_of_clusters


def _calculate_pairwise_distances(active_sites):
    """
    Calculates all pairwise distances between active sites

    Parameters
    ----------
    active_sites - list of ActiveSite instances

    Returns
    -------
    df - Pandas Dataframe containing condensed distance matrix
    """


    PDBs = [active_site.name for active_site in active_sites]
    df = pd.DataFrame(index=PDBs, columns=PDBs)
    for active_site_outer in range(len(active_sites)):
        for active_site_inner in range(len(active_sites[:active_site_outer])):
            df.ix[active_site_inner, active_site_outer] = compute_similarity(
                active_sites[active_site_outer].shell_matrix, active_sites[active_site_inner].shell_matrix)

    return df


def _plot_elbow_chart(dendrogram_list):
    """
    Plots an elbow chart with jump distances and acceleration

    Parameters
    ----------
    dendrogram_list

    Returns
    -------

    """
    sns.set_style("whitegrid")
    sns.despine()

    fig, ax = plt.subplots(figsize=(8, 8))
    cluster_jumps = [joining[2] for joining in reversed(dendrogram_list[-10:])]
    acceleration = np.diff(cluster_jumps, 2)
    steps = range(1, len(cluster_jumps) + 1)

    ax = plt.plot(steps, cluster_jumps)
    ax = plt.plot(steps[1:-1], acceleration)

    fig.savefig("JamesLucas_BMI203_HW2-Elbow.pdf", dpi=600, bbox_inches="tight")


def _plot_dendrogram(dendrogram_list, active_sites, cluster_indicies_dict):
    """
    Plots a dendrogram from the hierarchical clustering

    Parameters
    ----------
    dendrogram_list
    active_sites
    cluster_indicies_dict

    Returns
    -------
    """
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 8))

    adsf_double = np.array(dendrogram_list).astype("double")
    dendrogram_labels = [key for key in cluster_indicies_dict][:len(active_sites)]

    dn1 = hierarchy.dendrogram(adsf_double,
                               ax=ax,
                               above_threshold_color='y',
                               distance_sort=True,
                               orientation='left',
                               labels=dendrogram_labels
                               )

    fig.savefig("JamesLucas_BMI203_HW2-Dendrogram.pdf", dpi=600, bbox_inches="tight")

    return dn1


def evaluate_clusters_internally(clusterings, active_sites):
    """
    Implementing the Silhouette Index/Coefficient to determine the internal quality of my clusters
    http://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf

    Parameters
    ----------
    clusterings - List of lists containing PDB names
    active_sites - List of ActiveSite instances

    Returns
    -------
    None
    """

    # Initialize things
    df = _calculate_pairwise_distances(active_sites)
    silhouette_scores = []

    # Endless nested for loops...

    # For each active site in clusterings, calculate:
    # 1. Average distance of points within cluster (average_intra_cluster_distance)
    # 2. Average distance to another cluster (average_inter_cluster_distances)
    # 3. Silhouette coefficient ((average_inter_cluster_distance - average_intra_cluster_distance) / max(average_intra_cluster_distance, average_inter_cluster_distance))
    # Clustering_outer = internal cluster
    # clustering_inner = neighbor clusters

    # For each cluster...
    for index, clustering_outer in enumerate(clusterings):

        average_inter_cluster_distances = []

        # Iterate through all other clusters
        for same, clustering_inner in enumerate(clusterings):

            # Calculate average distance between all points in clustering_outer and clustering_inner
            if index != same:

                inter_cluster_distances = []

                for active_site_a in clustering_outer:
                    for active_site_b in clustering_inner:
                        if active_site_b > active_site_a:
                            inter_cluster_distances.append(df.ix[active_site_a, active_site_b]) # [row, column], distance
                        else:
                            inter_cluster_distances.append(df.ix[active_site_b, active_site_a]) # [row, column], distance

                average_inter_cluster_distances.append(sum(inter_cluster_distances) / len(inter_cluster_distances))

            # Calculate average intra-cluster distances
            else:

                intra_cluster_distances = []

                for offset, active_site_a in enumerate(clustering_outer):

                    for active_site_b in clustering_outer[:offset]:
                        if active_site_b > active_site_a:
                            intra_cluster_distances.append(df.ix[active_site_a, active_site_b])  # [row, column], distance
                        else:
                            intra_cluster_distances.append(df.ix[active_site_b, active_site_a])  # [row, column], distance

                if len(intra_cluster_distances) != 0:
                    average_intra_cluster_distance = sum(intra_cluster_distances) / len(intra_cluster_distances)

        # Calculate Silhouette scores for each (clustering_outer, clustering_inner) pair
        average_inter_cluster_distance = min(average_inter_cluster_distances)
        silhouette_scores.append((average_inter_cluster_distance - average_intra_cluster_distance) / max(average_intra_cluster_distance, average_inter_cluster_distance))

    final_silhouette_score = sum(silhouette_scores) / len(silhouette_scores)

    print(final_silhouette_score)

def compare_clusters(part_clusterings, hier_clusterings):
    """
    Calculate RAND score between clustering techniques
    Inspiration: http://stats.stackexchange.com/questions/89030/rand-index-calculation

    Parameters
    ----------
    part_clusterings
    hier_clusterings

    Returns
    -------
    rand_index
    """

    assert len(part_clusterings) == len(hier_clusterings), "Techniques need to output the same number of clusters!"

    # Generate co-occurrence matrix
    pre_matrix = []

    for p_index, p_cluster in enumerate(part_clusterings):
        current_row = []
        for h_index, h_cluster in enumerate(hier_clusterings):
            current_row.append(len(set(h_cluster) ^ set(p_cluster)))
        pre_matrix.append(current_row)

    co_occurance_matrix = np.asmatrix(pre_matrix)

    # Calculate TP, TN, FP, and FN

    column_sums = np.sum(co_occurance_matrix, axis = 0)
    tp_fp = np.sum([sp.misc.comb(value, 2) for value in column_sums])

    row_sums = np.sum(co_occurance_matrix, axis = 1)
    tp_fn = np.sum([sp.misc.comb(value, 2) for value in row_sums])

    tp_choose_two_matrix = sp.misc.comb(co_occurance_matrix, 2)
    tp = np.sum(tp_choose_two_matrix)

    fp = tp_fp - tp
    fn = tp_fn - tp

    co_occurance_matrix_sum = co_occurance_matrix.sum()
    tn = sp.misc.comb(co_occurance_matrix_sum, 2) - tp - fp - fn

    rand_index = (tp + tn) / (tp + tn + fp + fn)

    return rand_index





