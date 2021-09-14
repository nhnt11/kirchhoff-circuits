# @Author:  Felix Kramer <kramer>
# @Date:   2021-05-08T20:35:25+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-06-25T17:56:16+02:00
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
    kirchhoff_graph.set_source_landscape(mode='dipole_point')
    kirchhoff_graph.set_solute_landscape()

    kirchhoff_graph.scales['diffusion']=dict_pars['diffusion']
    kirchhoff_graph.scales['absorption']=dict_pars['absorption']
    kirchhoff_graph.set_absorption_landscape()
    kirchhoff_graph.set_geom_landscape()

    idx=np.where(kirchhoff_graph.nodes['solute'] > 0.)[0]
    kirchhoff_graph.scales['sum_flux']=np.sum(kirchhoff_graph.nodes['solute'][idx])

    return kirchhoff_graph

class flux_circuit(flow_circuit,object):

    def __init__(self):

        super(flux_circuit,self).__init__()

        self.nodes['solute']=[]
        self.nodes['concentration']=[]

        self.edges['peclet']=[]
        self.edges['absorption']=[]
        self.edges['length']=[]
        self.edges['radius']=[]
        self.edges['radius_sq']=[]
        self.edges['uptake']=[]

        self.scales.update({'flux':1})
        self.scales.update({'sum_flux':1})
        self.scales.update({'diffusion':1})
        self.scales.update({'absorption':1})

        self.graph.update({'solute_mode':''})
        self.graph.update({'absorption_mode':''})
        self.graph.update({'geom_mode':''})

        self.solute_mode={

            'default':self.init_solute_default,
            'custom':self.init_solute_custom
        }

        self.absorption_mode={

            'default':self.init_absorption_default,
            'random':self.init_absorption_random,
            'custom':self.init_absorption_custom
        }

        self.geom_mode={

            'default':self.init_geom_default,
            'random':self.init_geom_random,
            'custom':self.init_geom_custom
        }

    # set injection and outlet of metabolites
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

    # set spatial pattern of solute absorption rate
    def set_absorption_landscape(self, mode='default', **kwargs):

        # optional keywords
        if 'absorption' in kwargs:
            self.custom= kwargs['absorption']

        # call init sources
        if mode in self.absorption_mode.keys():

            self.absorption_mode[mode]()

        else :
            sys.exit('Whooops, Error: Define absorption rate pattern for the network.')

        self.graph['absorption_mode']=mode

    def init_absorption_default(self):

        self.edges['absorption']=np.ones(self.G.number_of_edges())*self.scales['absorption']

    def init_absorption_random(self):

        self.edges['absorption']=np.random.rand(self.G.number_of_edges())*self.scales['absorption']

    def init_absorption_custom(self):

        if len(self.custom.keys())==len(self.list_graph_edges):
            for j,edge in enumerate(self.list_graph_edges):

                c=self.custom[edge]*self.scales['absorption']
                self.edges['absorption'][j]=c
        else:

            print('Warning, custom absorption values ill defined, setting default !')
            self.init_absorption_default()

    # set spatial pattern of length and radii
    def set_geom_landscape(self, mode='default', **kwargs):

        # optional keywords
        if 'geom' in kwargs:
            self.custom= kwargs['geom']

        # call init sources
        if mode in self.geom_mode.keys():

            self.geom_mode[mode]()

        else :
            sys.exit('Whooops, Error: Define micro geometrics for the network.')

        self.graph['geom_mode']=mode

    def init_geom_default(self):

        self.edges['length']=np.ones(self.G.number_of_edges())*self.scales['length']

    def init_geom_random(self):

        self.edges['length']=np.random.rand(self.G.number_of_edges())*self.scales['length']

    def init_geom_custom(self,flux):

        if len(self.custom.keys())==len(self.list_graph_edges):
            for j,edge in enumerate(self.list_graph_edges):

                c=self.custom[edge]*self.scales['length']
                self.edges['length'][j]=c
        else:

            print('Warning, custom absorption values ill defined, setting default !')
            self.init_geom_default()


    def get_nodes_data(self):

        dn=pd.DataFrame(self.nodes[['source','solute','concentration']])

        return dn

    def get_edges_data(self, **kwargs):

        de=pd.DataFrame(self.edges[['conductivity','flow_rate','absorption','uptake','peclet','length']])

        if 'width' in kwargs:
            de['weight']=np.absolute(self.edges[kwargs['width']].to_numpy())*self.draw_weight_scaling
        else:
            de['weight']=np.power(self.edges['conductivity'].to_numpy(),0.25)*self.draw_weight_scaling

        return de
