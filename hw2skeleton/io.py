import glob
import os
from .utils import Atom, Residue, ActiveSite

# JAMES
import prody as pdy
import pandas as pd
import os

def read_active_sites(dir):
    """
    Read in all of the active sites from the given directory.

    Input: directory
    Output: list of ActiveSite instances
    """
    files = glob.glob(dir + '/*.pdb')

    active_sites = []
    # iterate over each .pdb file in the given directory
    for filepath in glob.iglob(os.path.join(dir, "*.pdb")):

        active_sites.append(read_active_site(filepath))

    print("Read in %d active sites"%len(active_sites))

    return active_sites


def read_active_site(filepath):
    """
    Read in a single active site given a PDB file

    Input: PDB file path
    Output: ActiveSite instance
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)

    if name[1] != ".pdb":
        raise IOError("%s is not a PDB file"%filepath)

    active_site = ActiveSite(name[0])

    r_num = 0

    # open pdb file
    with open(filepath, "r") as f:
        # iterate over each line in the file
        for line in f:
            if line[0:3] != 'TER':
                # read in an atom
                atom_type = line[13:17].strip()
                x_coord = float(line[30:38])
                y_coord = float(line[38:46])
                z_coord = float(line[46:54])
                atom = Atom(atom_type)
                atom.coords = (x_coord, y_coord, z_coord)

                residue_type = line[17:20]
                residue_number = int(line[23:26])

                # make a new residue if needed
                if residue_number != r_num:
                    residue = Residue(residue_type, residue_number)
                    r_num = residue_number

                # add the atom to the residue
                residue.atoms.append(atom)

            else:  # I've reached a TER card
                active_site.residues.append(residue)

    return active_site

def write_hierarchical_clustering(filename, clusters):
    """
    Outputs all hierarchical clusters created during clustering as a csv, where each column in a cluster
    This will include only non-singleton clusters

    Parameters
    ----------
    filename - filename from command line
    clusters - clusters_record OrderedDict

    Returns
    -------
    None
    """

    df = pd.DataFrame()
    for cluster in clusters:
        if len(clusters[cluster]) != 1:
            cluster_column = pd.Series(clusters[cluster], name=cluster)
            df = pd.concat([df, cluster_column], axis=1)

    df.to_csv(filename)
    print("It's dangerous to go alone! Take this: {}".format(filename))


def write_clustering(filename, clusters):
    """
    Outputs the final clusters created using a clustering algorithm as a csv, where each column in a cluster

    Parameters
    ----------
    filename - filename from command line
    clusters - list of lists

    Returns
    -------
    None
    """

    df = pd.DataFrame()
    cluster_count = 1

    for cluster in clusters:
        cluster_column = pd.Series(cluster, name="Cluster-{}".format(cluster_count))
        df = pd.concat([df, cluster_column], axis=1, ignore_index=True)
        cluster_count += 1

    df.to_csv(filename)
    print("I made dis: {}".format(filename))


def prody_import(pdb_directory):
    """
    Using prody to parse PDB files... makes it easier to get rid of duplicate
    active sites in asymmetric unit
    """

    parsed_pdb_list = []

    for pdb_file in os.listdir(pdb_directory):
        if pdb_file.lower().endswith(".pdb"):

            # Take the first chain of each PDB and name it after the PBD file
            current_pdb = pdy.parsePDB(os.path.join(pdb_directory, pdb_file))
            hv = pdy.HierView(current_pdb)

            active_site = ActiveSite(pdb_file.split('.')[0])
            active_site.residues = list(hv)[0]
            parsed_pdb_list.append(active_site)

    return parsed_pdb_list