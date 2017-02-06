import sys
from .cluster import cluster_by_partitioning, cluster_hierarchically, construct_active_site_matrix, evaluate_clusters_internally, compare_clusters

# JAMES
import pprint
from .io import prody_import, write_clustering

# Some quick stuff to make sure the program is called correctly
if len(sys.argv) < 5:
    print("Usage: python -m hw2skeleton [-P| -H] <pdb directory> <number of clusters> <output file>")
    sys.exit(0)

# active_sites = read_active_sites(sys.argv[2])
prody_active_sites = prody_import(sys.argv[2])
number = int(sys.argv[3])

# Choose clustering algorithm
if sys.argv[1][0:2] == '-P':
    print("Clustering using Partitioning method")
    construct_active_site_matrix(prody_active_sites)
    clusterings = cluster_by_partitioning(prody_active_sites, number)
    write_clustering(sys.argv[4], clusterings)
    evaluate_clusters_internally(clusterings, prody_active_sites)


if sys.argv[1][0:2] == '-H':
    print("Clustering using hierarchical method")
    construct_active_site_matrix(prody_active_sites)
    clusterings = cluster_hierarchically(prody_active_sites, number)
    write_clustering(sys.argv[4], clusterings)
    evaluate_clusters_internally(clusterings, prody_active_sites)

if sys.argv[1][0:2] == '-C':
    print("Comparing clustering methods using RAND index")
    construct_active_site_matrix(prody_active_sites)
    hier_clusterings = cluster_hierarchically(prody_active_sites, number)
    part_clusterings = cluster_by_partitioning(prody_active_sites, number)
    rand_index = compare_clusters(part_clusterings, hier_clusterings)
    print("Calculated RAND index for your two clusters: {}".format(rand_index))


