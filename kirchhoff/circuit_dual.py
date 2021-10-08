# @Author:  Felix Kramer
# @Date:   2021-05-22T13:11:37+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-06-22T23:01:04+02:00
# @License: MIT

import networkx as nx
import numpy as np
import scipy.linalg as lina
import sys

# custom embeddings/architectures for mono networkx
from kirchhoff.circuit_init import *
from kirchhoff.circuit_flow import *
from kirchhoff.circuit_flux import *

# custom primer
import kirchhoff.init_dual as init_dual

# custom output functions
import kirchhoff.draw_networkx as dx

def initialize_dual_circuit_from_networkx(input_graph1,input_graph2,e_adj):

    kirchhoff_dual_graph=dual_circuit()
    kirchhoff_dual_graph.circuit_init_from_networkx([input_graph1,input_graph2] )
    # kirchhoff_dual_graph=e_adj

    return kirchhoff_dual_graph

def initialize_dual_circuit_from_minsurf(dual_type='simple',num_periods=2):

    kirchhoff_dual_graph=dual_circuit()

    dual_graph=init_dual.init_dual_minsurf_graphs(dual_type,num_periods)

    kirchhoff_dual_graph.circuit_init_from_networkx([dual_graph.layer[0],dual_graph.layer[1]] )
    kirchhoff_dual_graph.distance_edges()

    return kirchhoff_dual_graph

def initialize_dual_circuit_from_catenation(dual_type='chain',num_periods=1):

    kirchhoff_dual_graph=dual_circuit()

    dual_graph=init_dual.init_dual_catenation(dual_type,num_periods)

    kirchhoff_dual_graph.circuit_init_from_networkx([dual_graph.layer[0],dual_graph.layer[1]] )

    return kirchhoff_dual_graph

def initialize_dual_flow_circuit_from_minsurf(dual_type='simple',num_periods=2):

    kirchhoff_dual_graph=dual_circuit()

    dual_graph=init_dual.init_dual_minsurf_graphs(dual_type,num_periods)

    kirchhoff_dual_graph.flow_circuit_init_from_networkx([dual_graph.layer[0],dual_graph.layer[1]] )
    kirchhoff_dual_graph.distance_edges()

    return kirchhoff_dual_graph

def initialize_dual_flux_circuit_from_minsurf(dual_type='simple',num_periods=2):

    kirchhoff_dual_graph=dual_circuit()

    dual_graph=init_dual.init_dual_minsurf_graphs(dual_type,num_periods)

    kirchhoff_dual_graph.flux_circuit_init_from_networkx([dual_graph.layer[0],dual_graph.layer[1]] )
    kirchhoff_dual_graph.distance_edges()

    return kirchhoff_dual_graph

class dual_circuit():

    def __init__(self):

        self.layer=[]
        self.e_adj=[]
        self.e_adj_idx=[]
        self.n_adj=[]

    def circuit_init_from_networkx(self, input_graphs ):

        self.layer=[]
        for G in input_graphs:

            self.layer.append(initialize_circuit_from_networkx(G))

    def flow_circuit_init_from_networkx(self, input_graphs ):

        self.layer=[]
        for G in input_graphs:

            self.layer.append(initialize_flow_circuit_from_networkx(G))

    def flux_circuit_init_from_networkx(self, input_graphs ):

        self.layer=[]
        for G in input_graphs:

            self.layer.append(initialize_flux_circuit_from_networkx(G))

    def distance_edges(self):

        self.D=np.zeros(len(self.e_adj_idx))
        for i,e in enumerate(self.e_adj_idx):

            n=self.layer[0].G.edges[e[0]]['slope'][0]-self.layer[0].G.edges[e[0]]['slope'][1]
            m=self.layer[1].G.edges[e[1]]['slope'][0]-self.layer[1].G.edges[e[1]]['slope'][1]
            q=np.cross(n,m)
            q/=np.linalg.norm(q)
            d=(self.layer[0].G.edges[e[0]]['slope'][0]-self.layer[1].G.edges[e[1]]['slope'][0])
            self.D[i]= np.linalg.norm(np.dot(d,q))

        self.D/=((self.layer[0].scales['length']+self.layer[1].scales['length'])/2.)

    def check_no_overlap(self,scale):

        check=True
        K1=self.layer[0]
        K2=self.layer[1]

        for e in self.e_adj:
            r1=K1.C[e[0],e[0]]
            r2=K2.C[e[1],e[1]]

            if r1+r2 > scale*0.5:
                check=False
                break
        return check

    def clipp_graph(self):
        for i in range(2):
            self.layer[i].clipp_graph()


    # output
    def plot_circuit(self,**kwargs):

        fig=dx.plot_networkx_dual(self,**kwargs)

        return fig
