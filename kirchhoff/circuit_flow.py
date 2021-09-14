# @Author:  Felix Kramer <kramer>
# @Date:   2021-05-08T20:35:25+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-09-13T22:57:38+02:00
# @License: MIT

import random as rd
import networkx as nx
import numpy as np
import sys
import pandas as pd

from kirchhoff.circuit_init import *

# custom embeddings/architectures
import kirchhoff.init_crystal as init_crystal
import kirchhoff.init_random as init_random

def initialize_flow_circuit_from_networkx(input_graph):

    kirchhoff_graph=flow_circuit()
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

def initialize_flow_circuit_from_crystal(crystal_type='default',periods=1):

    kirchhoff_graph=flow_circuit()
    input_graph=init_crystal.init_graph_from_crystal(crystal_type,periods)
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

def initialize_flow_circuit_from_random(random_type='default',periods=10,sidelength=1):

    kirchhoff_graph=flow_circuit()
    input_graph=init_random.init_graph_from_random(random_type,periods,sidelength)
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

def setup_default_flow_circuit(dict_pars):

    kirchhoff_graph=initialize_flow_circuit_from_networkx(dict_pars['skeleton'])
    kirchhoff_graph.set_source_landscape()
    kirchhoff_graph.set_plexus_landscape()

    return kirchhoff_graph

def setup_flow_circuit(dict_pars):

    kirchhoff_graph=initialize_flow_circuit_from_networkx(dict_pars['skeleton'])
    kirchhoff_graph.set_source_landscape(dict_pars['source'])
    kirchhoff_graph.set_plexus_landscape(dict_pars['plexus'])

    return kirchhoff_graph
# class flow_circuit(kirchhoff_init.circuit,object):
class flow_circuit(circuit,object):

    def __init__(self):

        super(flow_circuit,self).__init__()

        self.source_mode={

            'default':self.init_source_default,
            'root_geometric':self.init_source_root_central_geometric,
            'root_short':self.init_source_root_short,
            'root_long':self.init_source_root_long,
            'dipole_border':self.init_source_dipole_border,
            'dipole_point':self.init_source_dipole_point,
            'root_multi':self.init_source_root_multi,
            'custom':self.init_source_custom
        }

        self.plexus_mode={

            'default':self.init_plexus_default,
            'custom':self.init_plexus_custom,

        }


    # set a certain set of boundary conditions for the given networks
    def set_source_landscape(self,mode='default',**kwargs):

        # optional keywords
        if 'num_sources' in kwargs:
            self.graph['num_sources']= kwargs['num_sources']

        elif 'sources' in kwargs:
            self.custom= kwargs['sources']

        # else:
        #     print('Warning: Not recognizing certain keywords')
        # call init sources
        if mode in self.source_mode.keys():

            self.source_mode[mode]()

        else :
            sys.exit('Whooops, Error: Define Input/output-flows for the network.')

        self.graph['graph_mode']=mode
        self.test_source_consistency()

    def set_potential_landscape(self,mode):

        # todo
        return 0

    # different init source functions
    def init_source_custom(self):

        if len(self.custom.keys())==len(self.list_graph_nodes):

            for j,node in enumerate(self.list_graph_nodes):

                s=self.custom[node]*self.scales['flow']
                self.G.nodes[node]['source']=s
                self.nodes['source'][j]=s

        else:

            print('Warning, custom source values ill defined, setting default!')
            self.init_source_default()

    def init_source_default(self):

        centrality=nx.betweenness_centrality(self.G)
        centrality_sorted=sorted(centrality,key=centrality.__getitem__)

        self.set_root_leaves_relationship(centrality_sorted[-1])

    def init_source_root_central_geometric(self):

        pos=self.get_pos()
        X=np.mean(list(pos.values()),axis=0)

        dist={}
        for n in self.list_graph_nodes:
            dist[n]=np.linalg.norm(np.subtract(X,pos[n]))
        sorted_dist=sorted(dist,key=dist.__getitem__)

        self.set_root_leaves_relationship(sorted_dist[0])

    def init_source_root_short(self):

        # check whether geometric layout has been set
        pos=self.get_pos()

        # check for root closests to coordinate origin
        dist={}
        for n,p in pos.items():
            dist[n]=np.linalg.norm(p)
        sorted_dist=sorted(dist,key=dist.__getitem__)

        self.set_root_leaves_relationship(sorted_dist[0])

    def init_source_root_long(self):

        # check whether geometric layout has been set
        pos=self.get_pos()

        # check for root closests to coordinate origin
        dist={}
        for n,p in pos.items():
            dist[n]=np.linalg.norm(p)
        sorted_dist=sorted(dist,key=dist.__getitem__,reverse=True)

        self.set_root_leaves_relationship(sorted_dist[0])

    def init_source_dipole_border(self):

        pos=self.get_pos()
        dist={}
        for n,p in pos.items():
            dist[n]=np.linalg.norm(p[0])

        vals=list(dist.values())
        max_x=np.amax(vals)
        min_x=np.amin(vals)

        max_idx=[]
        min_idx=[]
        for k,v in dist.items():
            if v == max_x:
                max_idx.append(k)

            elif v == min_x:
                min_idx.append(k)

        self.set_poles_relationship(max_idx,min_idx)

    def init_source_dipole_point(self):

        pos=self.get_pos()
        dist={}
        for j,n in enumerate(self.list_graph_nodes[:-2]):
            for i,m in enumerate(self.list_graph_nodes[j+1:]):
                path=nx.shortest_path(self.G,source=n,target=m)
                dist[(n,m)]=len(path)
        max_len=np.amax(list(dist.values()))
        push=[]
        for key in dist.keys():
            if dist[key]==max_len:
                push.append(key)

        idx=np.random.choice(range(len(push)))
        source,sink=push[idx]

        self.set_poles_relationship([source],[sink])

    def init_source_root_multi(self):

        idx=np.random.choice( self.list_graph_nodes,size=self.graph['num_sources'] )
        self.nodes_source=[self.G.number_of_nodes()/self.graph['num_sources']-1,-1]

        for j,n in enumerate(self.list_graph_nodes):

            if n in idx:

                self.set_source_attributes(j,n,0)

            else:

                self.set_source_attributes(j,n,1)

    # auxillary function for the block above
    def set_root_leaves_relationship(self,root):

        self.nodes_source=[self.G.number_of_nodes()-1,-1]
        for j,n in enumerate(self.list_graph_nodes):

            if n==root:
                idx=0

            else:
                idx=1

            self.set_source_attributes(j,n,idx)

    def set_poles_relationship(self,sources,sinks):

        self.nodes_source=[1,-1,0]

        for j,n in enumerate(self.list_graph_nodes):
            self.set_source_attributes(j,n,2)

        for i,s in enumerate(sources):
            for j,n in enumerate(self.list_graph_nodes):
                if n==s:
                    self.set_source_attributes(j,s,0)

        for i,s in enumerate(sinks):
            for j,n in enumerate(self.list_graph_nodes):
                if n==s:
                    self.set_source_attributes(j,s,1)

    def set_source_attributes(self,j,node,idx):

        self.G.nodes[node]['source']=self.nodes_source[idx]*self.scales['flow']
        self.nodes['source'][j]=self.nodes_source[idx]*self.scales['flow']

    # different init potetnial functions
    def set_terminals_potentials(self,p0):
        idx_potential=[]
        idx_sources=[]
        for j,n in enumerate(nx.nodes(self.G)):

            if self.G.nodes[n]['source']>0:

                self.G.nodes[n]['potential']=1
                self.V[j]=p0
                idx_potential.append(j)
            elif self.G.nodes[n]['source']<0:

                self.G.nodes[n]['potential']=0.
                self.V[j]=0.
                idx_potential.append(j)
            else:

                self.G.nodes[n]['source']=0.
                self.J[j]=0.
                idx_sources.append(j)

        self.G.graph['sources']=idx_sources
        self.G.graph['potentials']=idx_potential

    # different init plexus functions

    def set_plexus_landscape(self,mode='default',**kwargs):

        # optional keywords

        if 'plexus' in kwargs:
            self.custom= kwargs['plexus']

        # call init sources
        if mode in self.plexus_mode.keys():

            self.plexus_mode[mode]()

        else :
            sys.exit('Whooops, Error: Define proper conductancies for  the network.')

        self.graph['plexus_mode']=mode
        self.test_conductance_consistency()

    def init_plexus_default(self):

        # find magnitude of flows and set scale of for conductancies
        d=np.amax(self.nodes['source']) * 0.5
        m=self.G.number_of_edges()
        self.edges['conductivity']=np.multiply(d,np.add(np.ones(m),np.random.rand(m)))

    def init_plexus_custom(self):

        if len(self.custom.keys())==len(self.list_graph_edges):
            # find magnitude of flows and set scale of for conductancies
            for j,edge in enumerate(self.list_graph_edges):

                c=self.custom[edge]*self.scales['conductance']
                self.G.edges[edge]['conductivity']=c
                self.edges['conductivity'][j]=c
        else:

            print('Warning, custom conductance values ill defined, setting default !')
            self.init_plexus_default()

    # output
    def get_nodes_data(self):

        dn=pd.DataFrame(self.nodes[['source','label']])

        return dn

    def get_edges_data(self,**kwargs):

        de=pd.DataFrame(self.edges[['conductivity','flow_rate']])

        if 'width' in kwargs:
            de['weight']=np.absolute(self.edges[kwargs['width']].to_numpy())*self.draw_weight_scaling
        else:
            de['weight']=np.power(self.edges['conductivity'].to_numpy(),0.25)*self.draw_weight_scaling

        # if 'color_edges' in pars:
        #     de['color_edges']=np.absolute(self.edges[pars['color']].to_numpy())
        # else:
        #     pass

        return de

    def plot_circuit(self, **kwargs):

        E=self.get_edges_data(**kwargs)
        V=self.get_nodes_data()

        self.set_pos()
        fig=dx.plot_networkx(   self.G, edge_list=self.list_graph_edges, node_list=self.list_graph_nodes, edge_data=E,  node_data=V )

        return fig
