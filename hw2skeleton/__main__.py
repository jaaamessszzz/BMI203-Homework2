import sys
from .cluster import cluster_by_partitioning, cluster_hierarchically, evaluate_clusters_internally, compare_clusters, compute_similarity

# JAMES
import pprint
from .io import prody_import, write_clustering
import os

# Some quick stuff to make sure the program is called correctly
if len(sys.argv) < 5:
    print("Usage: python -m hw2skeleton [-P| -H| -C] <pdb directory> <number of clusters> <output file>")
    sys.exit(0)

prody_active_sites = prody_import(sys.argv[2])
number = int(sys.argv[3])


# Choose clustering algorithm
if sys.argv[1][0:2] == '-P':
    print("Clustering using Partitioning method")
    clusterings = cluster_by_partitioning(prody_active_sites, number)
    write_clustering("Part", sys.argv[4], clusterings)

    pprint.pprint(clusterings)
    evaluate_clusters_internally(clusterings, prody_active_sites)

if sys.argv[1][0:2] == '-H':
    print("Clustering using hierarchical method")
    clusterings = cluster_hierarchically(prody_active_sites, number)
    write_clustering("Hier", sys.argv[4], clusterings)
    evaluate_clusters_internally(clusterings, prody_active_sites)

if sys.argv[1][0:2] == '-C':
    print("Comparing clustering methods using RAND index")
    hier_clusterings = cluster_hierarchically(prody_active_sites, number)
    part_clusterings = cluster_by_partitioning(prody_active_sites, number)
    write_clustering("Hier", sys.argv[4], hier_clusterings)
    write_clustering("Part", sys.argv[4], part_clusterings)
    rand_index = compare_clusters(part_clusterings, hier_clusterings)
    print("Calculated RAND index for your two clusters: {}".format(rand_index))


