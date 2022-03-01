# @Author:  Felix Kramer
# @Date:   2021-05-22T13:11:37+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-07T16:06:52+01:00
# @License: MIT

import networkx as nx
import numpy as np
import scipy.linalg as lina
from dataclasses import dataclass, field
# custom embeddings/architectures for mono networkx
from kirchhoff.circuit_init import *
from kirchhoff.circuit_flow import *
from kirchhoff.circuit_flux import *

# custom primer
import kirchhoff.init_dual as init_dual

# custom output functions
import kirchhoff.draw_networkx as dx


def construct_from_graphSet(graphSet, circuit_type):

    circuitConstructor = {
        'default' : Circuit,
        'circuit' : Circuit,
        'flow' : FlowCircuit,
        'flux' : FluxCircuit,
        }

    circuitSet = []
    if circuit_type in circuitConstructor:

        for g in graphSet.layer:
            K = circuitConstructor[circuit_type](g)

            circuitSet.append(K)

    else:
        print('Warning: Invalid graph mode, choose default')

        for g in graphSet.layer:
            K = circuitConstructor['default'](g)
            circuitSet.append(K)

    return circuitSet

def initialize_dual_from_catenation(dual_type='catenation', num_periods=1, circuit_type='default'):

    """
    Initialize a dual spatially embedded circuit, with internal graphs based on
    simple catenatednetwork skeletons.

    Args:
        dual_type (string): The type of dual skeleton (simple, diamond, laves, catenation).
        num_periods (int): Repetition number of the lattice's unit cell.

    Returns:
        dual_circuit: A dual circuit system.

    """

    graphSet = init_dual.init_dualCatenation(dual_type, num_periods)
    circuitSet = construct_from_graphSet(graphSet, circuit_type)
    for i, g in enumerate(graphSet.layer):
        circuitSet[i].info = circuitSet[i].set_info(g, dual_type)

    kirchhoff_dual = DualCircuit(circuitSet)

    return kirchhoff_dual

def initialize_dual_from_minsurf(dual_type='simple', num_periods=2, circuit_type='default'):
    """
    Initialize a dual spatially embedded flux circuit, with internal graphs based on
     the network skeletons of triply-periodic minimal surfaces.

    Args:
        dual_type (string): The type of dual skeleton (simple, diamond, laves, catenation).
        num_periods (int): Repetition number of the lattice's unit cell.

    Returns:
        dual_circuit: A flow_circuit object.

    """

    graphSet = init_dual.init_dual_minsurf_graphs(dual_type, num_periods)
    circuitSet = construct_from_graphSet(graphSet, circuit_type)
    for i, g in enumerate(graphSet.layer):
        circuitSet[i].info = circuitSet[i].set_info(g, dual_type)
    kirchhoff_dual = DualCircuit(circuitSet)

    kirchhoff_dual.e_adj = graphSet.e_adj
    kirchhoff_dual.e_adj_idx = graphSet.e_adj_idx
    kirchhoff_dual.distance_edges()

    return kirchhoff_dual

@dataclass
class DualCircuit():

    """
    A base class for flow circuits.

    Attributes
    ----------
        layer (list): List of the graphs contained in the multilayer circuit.
        e_adj (list): A list off edge affiliation between the different layers, edge view.
        e_adj_idx (list): A list off edge affiliation between the different layers, label view.
        n_adj (list): An internal nodal varaible.
    """

    layer: list = field(default_factory=list, repr=False)
    e_adj: list = field(default_factory=list, repr=False)
    e_adj_idx: list = field(default_factory=list, repr=False)
    n_adj: list = field(default_factory=list, repr=False)

    def distance_edges(self):

        """
        Compute the distance of affiliated edges in the multilayer circuit.

        """

        self.dist_adj = np.zeros(len(self.e_adj_idx))

        for i, e in enumerate(self.e_adj_idx):

            g1 = self.layer[0].G
            pos1 = [g1.nodes[node]['pos'] for node in e[0]]
            p1 = np.mean(pos1,axis = 0)


            g2 = self.layer[1].G
            pos2 = [g2.nodes[node]['pos'] for node in e[1]]
            p2 = np.mean(pos2,axis = 0)

            self.dist_adj[i] = np.linalg.norm(p1-p2)

    # def check_no_overlap(self, scale):
    #
    #     """
    #     Test whether the multilayer systems have geometrically overlapping edges.
    #
    #     Returns:
    #         bool: True if systems geometrically overlap, otherwise False
    #
    #     """
    #
    #
    #     check = True
    #     K1 = self.layer[0]
    #     K2 = self.layer[1]
    #
    #     for e in self.e_adj:
    #         r1 = K1.C[e[0], e[0]]
    #         r2 = K2.C[e[1], e[1]]
    #
    #         if r1+r2 > scale*0.5:
    #             check = False
    #             break
    #
    #     return check

    # def clipp_graph(self):
    #
    #     """
    #     Prune the internal graph variables, using an edge weight threshold criterium.
    #     """
    #
    #     for i in range(2):
    #         self.layer[i].clipp_graph()

    # output
    def plot_circuit(self, *args, **kwargs):

        """
        Use Plotly.GraphObjects to create interactive plots that have
         optionally the graph atributes displayed.
        Args:
            kwargs (dictionary): A dictionary for plotly keywords customizing the plots' layout.

        Returns:
            plotly.graph_objects.Figure: A plotly figure displaying the circuit.

        """
        fig = dx.plot_networkx_dual(self, *args, **kwargs)

        return fig
