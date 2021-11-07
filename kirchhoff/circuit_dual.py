# @Author:  Felix Kramer
# @Date:   2021-05-22T13:11:37+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-07T12:46:09+01:00
# @License: MIT

import networkx as nx
import numpy as np
import scipy.linalg as lina
# custom embeddings/architectures for mono networkx
from kirchhoff.circuit_init import *
from kirchhoff.circuit_flow import *
from kirchhoff.circuit_flux import *

# custom primer
import kirchhoff.init_dual as init_dual

# custom output functions
import kirchhoff.draw_networkx as dx

def initialize_dual_circuit_from_networkx(input_graph1, input_graph2, e_adj):

    kirchhoff_dual = dual_circuit()
    kirchhoff_dual.circuit_init_from_networkx([input_graph1, input_graph2])
    # kirchhoff_dual_graph = e_adj

    return kirchhoff_dual

def initialize_dual_circuit_from_minsurf(dual_type='simple', num_periods=2):

    kirchhoff_dual = dual_circuit()

    dual_graph = init_dual.init_dual_minsurf_graphs(dual_type, num_periods)

    kirchhoff_dual.circuit_init_from_networkx([g for g in dual_graph.layer])
    kirchhoff_dual.distance_edges()

    return kirchhoff_dual

def initialize_dual_circuit_from_catenation(dual_type='catenation', num_periods=1):

    kirchhoff_dual = dual_circuit()

    dual_graph = init_dual.init_dual_catenation(dual_type, num_periods)

    kirchhoff_dual.circuit_init_from_networkx([g for g in dual_graph.layer])

    return kirchhoff_dual

def initialize_dual_flow_circuit_from_minsurf(dual_type='simple', num_periods=2):

    kirchhoff_dual = dual_circuit()

    dual_graph = init_dual.init_dual_minsurf_graphs(dual_type, num_periods)

    kirchhoff_dual.flow_circuit_init_from_networkx([g for g in dual_graph.layer])
    kirchhoff_dual.distance_edges()

    return kirchhoff_dual

def initialize_dual_flux_circuit_from_minsurf(dual_type='simple', num_periods=2):

    kirchhoff_dual = dual_circuit()

    dual_graph = init_dual.init_dual_minsurf_graphs(dual_type, num_periods)

    kirchhoff_dual.flux_circuit_init_from_networkx([g for g in dual_graph.layer])
    kirchhoff_dual.distance_edges()

    return kirchhoff_dual

def initialize_dual_flow_circuit_from_catenation(dual_type='catenation', num_periods=1):

    kirchhoff_dual = dual_circuit()

    dual_graph = init_dual.init_dual_catenation(dual_type, num_periods)

    kirchhoff_dual.flow_circuit_init_from_networkx([g for g in dual_graph.layer])

    return kirchhoff_dual

def initialize_dual_flux_circuit_from_catenation(dual_type='catenation', num_periods=1):

    kirchhoff_dual = dual_circuit()

    dual_graph = init_dual.init_dual_catenation(dual_type, num_periods)

    kirchhoff_dual.flux_circuit_init_from_networkx([g for g in dual_graph.layer])

    return kirchhoff_dual

def initialize_dual_flow_circuit_from_networkx(input_graph1, input_graph2, e_adj):

    kirchhoff_dual = dual_circuit()
    kirchhoff_dual.flow_circuit_init_from_networkx([input_graph1, input_graph2])

    return kirchhoff_dual

def initialize_dual_flux_circuit_from_networkx(input_graph1, input_graph2, e_adj):

    kirchhoff_dual = dual_circuit()
    kirchhoff_dual.flux_circuit_init_from_networkx([input_graph1, input_graph2])

    return kirchhoff_dual

class dual_circuit():

    def __init__(self):

        self.layer = []
        self.e_adj = []
        self.e_adj_idx = []
        self.n_adj = []

    def circuit_init_from_networkx(self, input_graphs):

        self.layer = []
        for G in input_graphs:

            self.layer.append(initialize_circuit_from_networkx(G))

    def flow_circuit_init_from_networkx(self,  input_graphs ):

        self.layer = []
        for G in input_graphs:

            self.layer.append(initialize_flow_circuit_from_networkx(G))

    def flux_circuit_init_from_networkx(self,  input_graphs ):

        self.layer = []
        for G in input_graphs:

            self.layer.append(initialize_flux_circuit_from_networkx(G))

    def distance_edges(self):

        self.D = np.zeros(len(self.e_adj_idx))
        for i, e in enumerate(self.e_adj_idx):

            p1 = [q for q in self.layer[0].G.edges[e[0]]['slope']]
            n = p1[0]-p1[1]

            p2 = [q for q in self.layer[0].G.edges[e[0]]['slope']]
            m = p2[0]-p2[1]

            q = np.cross(n, m)
            q /= np.linalg.norm(q)

            c = [self.layer[0].G.edges[u]['slope'][0] for u in e]
            d = c[0]-c[1]
            self.D[i] = np.linalg.norm(np.dot(d, q))

        s1 = self.layer[0].scales['length']
        s2 = self.layer[1].scales['length']
        self.D /= ((s1+s2)/2.)

    def check_no_overlap(self, scale):

        check = True
        K1 = self.layer[0]
        K2 = self.layer[1]

        for e in self.e_adj:
            r1 = K1.C[e[0], e[0]]
            r2 = K2.C[e[1], e[1]]

            if r1+r2 > scale*0.5:
                check = False
                break

        return check

    def clipp_graph(self):
        for i in range(2):
            self.layer[i].clipp_graph()

    # output
    def plot_circuit(self, **kwargs):

        fig = dx.plot_networkx_dual(self, **kwargs)

        return fig
