from .utils import Atom, Residue, ActiveSite

# JAMES
import prody as pd
import numpy as np
from .utils import ClusterPartition
import pandas as pd

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

    print(site_a)
    print(site_b)
    print(similarity)

    return similarity

def construct_active_site_matrix(active_sites):
    """
    This function calculates the atom count matrix for each active site.

    To calculate the matrix:
    1. The centroid of the active site is calculated


    Parameters
    ----------
    active_sites: list of active site objects

    Returns
    -------

    """
    for active_site in active_sites:

        # df index = step // 0.5
        steps = [np.arange(0, 21, 1)]
        df = pd.DataFrame(np.zeros((21, 4)), index=steps, columns=['C', 'O', 'N', 'S'])

        # Calculate centroid
        active_site_coords = active_site.residues.getCoords()

        length = active_site_coords.shape[0]
        sum_x = np.sum(active_site_coords[:, 0])
        sum_y = np.sum(active_site_coords[:, 1])
        sum_z = np.sum(active_site_coords[:, 2])

        active_site_center = np.array([sum_x/length, sum_y/length, sum_z/length])

        # Populate dataframe with atom types and distances
        # Step size of 0.5 in cartesian space... not sure how that translates to Angstroms
        for residue in active_site.residues:
            for atom in residue:
                distance = np.linalg.norm(atom.getCoords() - active_site_center)
                if distance // 0.5 >= 20:
                    shell = 20
                else:
                    shell = distance // 0.5

                # Save distance in matrix if Atom type is recognized
                # Some of the PDBs still have hydrogens in them...
                if atom.getName()[0] in "CONS":
                    df.ix[shell, atom.getName()[0]] += 1

        active_site.shell_matrix = np.asmatrix(df.values)


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

    # Select random active site coordinates to serve as initial cluster centers
    cluster_centers = []
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
            print("Min distance: {} - {}".format(min_distance_cluster, calculated_distances[min_distance_cluster]))

            for center in current_clusters:
                if center.cluster_ID == min_distance_cluster:
                    center.cluster_members_current.append(active_site)
                    break

        # Update cluster centers
        stop_counter = 0

        for center in current_clusters:
            center.cluster_center = np.mean(np.array([active_site.shell_matrix for active_site in center.cluster_members_current]), axis=0)

            # print("cluster_members_previous")
            # print(center.cluster_members_previous)
            # print("cluster_members_current")
            # print(center.cluster_members_current)

            # Update Break condition
            if center.cluster_members_previous == center.cluster_members_current:
                stop_counter += 1
                # print("Stop counter incremented!")

            center.cluster_members_previous = center.cluster_members_current

        # Break Condition
        if stop_counter == len(current_clusters):
            stop = True

    print("Termination criteria reached after {} iterations".format(stop))

    return [cluster.cluster_members_current for cluster in current_clusters]

def cluster_hierarchically(active_sites):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.

    I need a bunch of different matricies to record data for this clustering... following scikitlearn:
    1. A distance matrix (implemented as pandas dataframe... clunky but it will work)
    2. A 4 x N matrix to record which clusters I join together, their distance, and how many members are in the cluster
    3. ????

    I need to update the distance matrix every time I create a new cluster. One of the reasons I opted to go with the
    pandas dataframe in lieu of a numpy array is that I can use index labels instead of needing to keep track of indices
    every time I update the matrix where things will move around.

    I am recording cluster joinings in a 4 x N matrix so that I can use the scikitlearn dendrogram at the end of this to
    evaluate my clusterings

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """

    # Calculate all pairwise distances between active sites
    # Distance matrix represented in a Pandas Dataframe
    PDBs = [active_site.name for active_site in active_sites]
    df = pd.DataFrame(index=PDBs, columns=PDBs)
    for active_site_outer in range(len(active_sites)):
        for active_site_inner in range(len(active_sites[:active_site_outer])):
            df.ix[active_site_inner, active_site_outer] = compute_similarity(active_sites[active_site_outer].shell_matrix, active_sites[active_site_inner].shell_matrix)

    # Output for debugging
    df.to_csv("FUCK.csv")

    # Find the minimum distance between two clusters (initially singletons)

    # Combine the two clusters, record in 4 x N matrix





    return []
