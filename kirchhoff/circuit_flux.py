# @Author:  Felix Kramer <kramer>
# @Date:   2021-05-08T20:35:25+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-05-23T23:55:42+02:00
# @License: MIT

import random as rd
import networkx as nx
import numpy as np
import sys
import pandas as pd

from kirchhoff.circuit_flow import *
import kirchhoff.init_crystal as init_crystal
import kirchhoff.init_random as init_random

def initialize_flux_circuit_from_networkx(input_graph):

    kirchhoff_graph=flux_circuit()
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

def initialize_flux_circuit_from_random(random_type='default',periods=10,sidelength=1):

    kirchhoff_graph=flux_circuit()
    input_graph=init_random.init_graph_from_random(random_type,periods,sidelength)
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

def initialize_flux_circuit_from_crystal(crystal_type='default',periods=1):

    kirchhoff_graph=flux_circuit()
    input_graph=init_crystal.init_graph_from_crystal(crystal_type,periods)
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

def setup_default_flux_circuit(dict_pars):

    kirchhoff_graph=initialize_flux_circuit_from_networkx(dict_pars['plexus'])
    kirchhoff_graph.set_solute_landscape()

    self.scales['diffusion']=dict_pars['diffusion']
    self.scales['absorption']=dict_pars['absorption']
    idx=np.where(self.nodes['solute'] > 0.)
    self.scales['sum_flux']=np.sum(self.nodes['solute'][idx])


# class flux_circuit(kirchhoff_flow.flow_circuit,object):
class flux_circuit(flow_circuit,object):
    def __init__(self):

        super(flux_circuit,self).__init__()

        self.nodes['solute']=[]
        self.nodes['concentration']=[]

        self.scales.update({'flux':1})
        self.scales.update({'sum_flux':1})
        self.scales.update({'diffusion':1})
        self.scales.update({'absorption':1})
        self.graph.update({'solute_mode':''})

        self.solute_mode={

            'default':self.init_solute_default,
            'custom':self.init_solute_custom
        }

    def set_solute_landscape(self, mode='default', **kwargs):

        # optional keywords
        if 'solute' in kwargs:
            self.custom= kwargs['solute']


        # call init sources
        if mode in self.solute_mode.keys():

            self.solute_mode[mode]()

        else :
            sys.exit('Whooops, Error: Define Input/output-flows for the network.')

        self.graph['solute_mode']=mode

    def init_solute_default(self):

        vals=[1,-1,0]

        for j,n in enumerate(self.list_graph_nodes):

            if self.nodes['source'][j] > 0:
                self.set_solute(j,n,vals[0])

            elif self.nodes['source'][j] < 0:
                self.set_solute(j,n,vals[1])

            else:
                self.set_solute(j,n,vals[2])

    def init_solute_custom(self,flux):

        if len(self.custom.keys())==len(self.list_graph_nodes):

            for j,node in enumerate(self.list_graph_nodes):

                f=self.custom[node]*self.scales['flux']
                self.nodes['solute'][j]=f
                self.G.nodes[node]['solute']=f

        else:

            print('Warning, custom solute values ill defined, setting default!')
            self.init_solute_default()

    def set_solute(self,idx,nodes,vals):

        f=self.scales['flux']*vals
        self.nodes['solute'][idx]=f
        self.G.nodes[nodes]['solute']=f
