# Some utility classes to represent a PDB structure

class Atom:
    """
    A simple class for an amino acid residue
    """

    def __init__(self, type):
        self.type = type
        self.coords = (0.0, 0.0, 0.0)

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return self.type

class Residue:
    """
    A simple class for an amino acid residue
    """

    def __init__(self, type, number):
        self.type = type
        self.number = number
        self.atoms = []

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return "{0} {1}".format(self.type, self.number)

class ActiveSite:
    """
    A simple class for an active site
    """

    def __init__(self, name):
        self.name = name
        self.residues = []
        self.resnums = 0
        self.atom_C = 0
        self.atom_O = 0
        self.atom_N = 0
        self.atom_S = 0

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return self.name

# ClusterPartition class for keeping track of cluster centers and cluster members
class ClusterPartition(object):
    def __init__(self):
        self.cluster_center = None
        self.cluster_members = []

    # Calculate the cluster center based on class members and save to cluster_center
    def calculate_center(self):
        pass