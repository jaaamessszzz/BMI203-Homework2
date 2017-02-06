from hw2skeleton import cluster
from hw2skeleton import io
import os

def test_similarity():
    filename_a = os.path.join("./data", "276.pdb")
    filename_b = os.path.join("./data", "4629.pdb")

    activesite_a = io.prody_import(filename_a)
    activesite_b = io.prody_import(filename_b)

    print(activesite_a)
    print(activesite_b)

    print(cluster.compute_similarity(activesite_a.shell_matrix, activesite_b.shell_matrix))

    # update this assertion
    assert cluster.compute_similarity(activesite_a.shell_matrix, activesite_b.shell_matrix) == 21.42428528562855

def test_partition_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("./data", "%i.pdb" % id)
        active_sites.append(io.prody_import(filepath))

    clusterings = cluster.cluster_by_partitioning(active_sites, 3)

    # Organize clusterings since they get spit out randomly...
    numbers = sorted([int(out[0]) for out in clusterings])
    test_clusterings = [[str(label)] for label in numbers]

    # update this assertion
    assert test_clusterings == [['276'], ['4629'], ['10701']]


def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("./data", "%i.pdb"%id)
        active_sites.append(io.prody_import(filepath))

    clusterings = cluster.cluster_hierarchically(active_sites, 3)

    # Organize clusterings since they get spit out randomly...
    numbers = sorted([int(out[0]) for out in clusterings])
    test_clusterings = [[str(label)] for label in numbers]

    # update this assertion
    assert clusterings == [['276'], ['4629'], ['10701']]

def test_evaluate_clusters_internally():
    sites = [['276.pdb', '1806.pdb', '41719.pdb', '41729.pdb']]
    prody_active_sites = []

    for site in sites:
        prody_active_sites.append(io.prody_import(os.path.join('./data', site)))

    test_clusters = [['276', '1806'], ['41719', '41729']]
    score = cluster.evaluate_clusters_internally(test_clusters, prody_active_sites)

    print(score)
    assert score == 0.454018769612
