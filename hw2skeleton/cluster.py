from .utils import Atom, Residue, ActiveSite

# JAMES
import prody as pd
from Bio import pairwise2
import numpy as np
from .utils import ClusterPartition
import pprint
import sys

def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """

    similarity = 0.0

    # Fill in your code here!

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
    4. Back to 2. until termination criteria are met

    Similarity Metric: Distance between two vectors of normalized active site attributes

    Metrics:
    # number of residues: Self explanatory
    # atom types vector [C, O, N, S]: This will give me an idea of residue types in the active site
    # average interatomic distances: How large is the active site?

    Ideally would have wanted to use RMSD, but there were a lot of difficulties


    """


    ###########################################
    # Calculate all required metrics and save #
    ###########################################

    # Number of Residues
    resnum_dict = {active_site.name: active_site.residues.numResidues() for active_site in active_sites}
    max_resnum = max(resnum_dict, key=(lambda key: resnum_dict[key]))
    print("Max resnum: {} - {}".format(max_resnum, resnum_dict[max_resnum]))

    # Atom Types
    for active_site in active_sites:
        atom_list = []
        for residue in active_site.residues:
            for atom in residue:
                atom_list.append(atom.getName()[0])

        active_site.atom_C = atom_list.count("C")
        active_site.atom_O = atom_list.count("O")
        active_site.atom_N = atom_list.count("N")
        active_site.atom_S = atom_list.count("S")

        # print(active_site.name)
        # print(active_site.residues.getSequence())
        # print(atom_list)

    max_atoms_C_dict = {active_site.name: active_site.atom_C for active_site in active_sites}
    max_atoms_O_dict = {active_site.name: active_site.atom_O for active_site in active_sites}
    max_atoms_N_dict = {active_site.name: active_site.atom_N for active_site in active_sites}
    max_atoms_S_dict = {active_site.name: active_site.atom_S for active_site in active_sites}

    max_C = max(max_atoms_C_dict, key=(lambda key: max_atoms_C_dict[key]))
    max_O = max(max_atoms_O_dict, key=(lambda key: max_atoms_O_dict[key]))
    max_N = max(max_atoms_N_dict, key=(lambda key: max_atoms_N_dict[key]))
    max_S = max(max_atoms_S_dict, key=(lambda key: max_atoms_S_dict[key]))

    print("Max C: {} - {}".format(max_C, max_atoms_C_dict[max_C]))
    print("Max O: {} - {}".format(max_O, max_atoms_O_dict[max_O]))
    print("Max N: {} - {}".format(max_N, max_atoms_N_dict[max_N]))
    print("Max S: {} - {}".format(max_S, max_atoms_S_dict[max_S]))

    sys.exit()

    # Distance

    distance_dict = {}
    edge_count = 0
    distance = 0

    for active_site in active_sites:
        for index_i, i in enumerate(active_site.residues):
            for index_j, j in enumerate(active_site.residues[:index_i]):
                edge_count += 1
                distance += np.linalg.norm(i-j)
        distance_dict[active_site.name] = distance / edge_count

    max_distance = max(distance_dict, key=(lambda key: distance_dict[key]))
    print("Max distance: {}".format(max_distance))

    # Sequence
    sequence_dict = {active_site.name: active_site.residues.getSequence() for active_site in active_sites}

    ########################################
    # Implement Clustering by Partitioning #
    ########################################

    # Select random active site coordinates to serve as initial cluster centers
    cluster_centers = []
    while len(cluster_centers) != cluster_number:
        active_site_pick = active_sites[np.random.randint(0, len(active_sites))]
        if active_site_pick not in cluster_centers:
            cluster_centers.append(active_site_pick)

    # Instantiate cluster centers
    current_clusters = []
    for cluster_center in cluster_centers:
        K_mean = ClusterPartition()
        # Calculate initial metrics
        initial_resnum = resnum_dict[cluster_center] / max_resnum
        initial_distance = distance_dict[cluster_center] / max_distance
        initial_sequence = sequence_dict[cluster_center]

        K_mean.cluster_center = (initial_resnum, initial_distance, initial_sequence)
        current_clusters.append(K_mean)

    # Iterate through active sites and assign to clusters
    for active_site in active_sites:

        high_score = (None, 0)

        for center in current_clusters:
            #

            alignments = pairwise2.align.globalxx(center.cluster_center.getSequence(), active_site.getSequence(), score_only=True)
            if high_score[1] <= alignments:
                high_score = (center, alignments)
                print("Updating high score: {} - {}".format(center, alignments))

        high_score[0].cluster_members.append(active_site)

    pprint.pprint([cluster_center.cluster_members for cluster_center in current_clusters])

    # Update cluster centers
    # for center in current_clusters:

    return [cluster_center.cluster_members for cluster_center in current_clusters]


def cluster_hierarchically(active_sites):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """

    # Fill in your code here!

    return []
