import glob
import os
from .utils import Atom, Residue, ActiveSite

# JAMES
import prody as pdy
import numpy as np
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


def write_clustering(clustering_type, filename, clusters):
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

    df.to_csv("-".join([clustering_type, filename]))
    print("I made dis: {}".format(filename))


def prody_import(pdb_input):
    """
    Using prody to parse PDB files... makes it easier to get rid of duplicate
    active sites in asymmetric unit
    """

    if os.path.isfile(pdb_input):
        print("Parsing {} as a file".format(pdb_input))
        current_pdb = pdy.parsePDB(os.path.join(pdb_input))
        hv = pdy.HierView(current_pdb)

        active_site = ActiveSite(os.path.split(pdb_input)[1].split('.')[0])
        active_site.residues = list(hv)[0]

        construct_active_site_matrix([active_site])

        return active_site

    elif os.path.isdir(pdb_input):
        print("Parsing {} as a directory".format(pdb_input))

        parsed_pdb_list = []

        for pdb_file in os.listdir(pdb_input):
            if pdb_file.lower().endswith(".pdb"):

                # Take the first chain of each PDB and name it after the PBD file
                current_pdb = pdy.parsePDB(os.path.join(pdb_input, pdb_file))
                hv = pdy.HierView(current_pdb)

                active_site = ActiveSite(pdb_file.split('.')[0])
                active_site.residues = list(hv)[0]
                parsed_pdb_list.append(active_site)

        construct_active_site_matrix(parsed_pdb_list)

        return parsed_pdb_list

def construct_active_site_matrix(active_sites):
    """
    This function calculates the atom count matrix for each active site.

    To calculate the matrix:
    1. The centroid of the active site is calculated
    2. Calcualte the distance of each atom from the centroid
    3. Populate the dataframe by step size and atom type
    4. Assign DataFrame as numpy matrix to active_site.shell_matrix

    Parameters
    ----------
    active_sites: list of active site objects

    Returns
    -------
    None
    """
    # Settings!
    number_of_shells = 20
    step_size = 0.2


    for active_site in active_sites:

        # df index = step // step_size
        steps = [np.arange(0, number_of_shells, 1)]
        df = pd.DataFrame(np.zeros((number_of_shells, 4)), index=steps, columns=['C', 'O', 'N', 'S'])

        # Calculate centroid
        active_site_coords = active_site.residues.getCoords()

        length = active_site_coords.shape[0]
        sum_x = np.sum(active_site_coords[:, 0])
        sum_y = np.sum(active_site_coords[:, 1])
        sum_z = np.sum(active_site_coords[:, 2])

        active_site_center = np.array([sum_x/length, sum_y/length, sum_z/length])

        # Populate dataframe with atom types and distances
        # Step size in cartesian space... not sure how that translates to Angstroms
        for residue in active_site.residues:
            for atom in residue:

                distance = np.linalg.norm(atom.getCoords() - active_site_center)

                if distance // step_size >= number_of_shells - 1:
                    shell = number_of_shells - 1
                else:
                    shell = distance // step_size

                # Save distance in matrix if Atom type is recognized
                # Some of the PDBs still have hydrogens in them...
                if atom.getName()[0] in "CONS":
                    df.ix[shell, atom.getName()[0]] += 1

        active_site.shell_matrix = np.asmatrix(df.values)