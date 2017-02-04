import sys
from .io import read_active_sites, write_clustering, write_mult_clusterings
from .cluster import cluster_by_partitioning, cluster_hierarchically, construct_active_site_matrix

# JAMES
import pprint
from .io import prody_import

# Some quick stuff to make sure the program is called correctly
if len(sys.argv) < 4:
    print("Usage: python -m hw2skeleton [-P| -H] <pdb directory> <output file>")
    sys.exit(0)

# active_sites = read_active_sites(sys.argv[2])
prody_active_sites = prody_import(sys.argv[2])

# Development
if sys.argv[1][0:2] == '-D':
    for number in range(5,6):
        construct_active_site_matrix(prody_active_sites)
        asdf = cluster_by_partitioning(prody_active_sites, number)
        pprint.pprint(asdf)
        break

# Choose clustering algorithm
if sys.argv[1][0:2] == '-P':
    print("Clustering using Partitioning method")
    construct_active_site_matrix(prody_active_sites)
    clustering = cluster_by_partitioning(prody_active_sites)
    write_clustering(sys.argv[3], clustering)

if sys.argv[1][0:2] == '-H':
    print("Clustering using hierarchical method")
    construct_active_site_matrix(prody_active_sites)
    clusterings = cluster_hierarchically(prody_active_sites)
    # write_mult_clusterings(sys.argv[3], clusterings)
