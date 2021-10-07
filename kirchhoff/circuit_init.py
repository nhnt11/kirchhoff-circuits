# @Author:  Felix Kramer <kramer>
# @Date:   2021-05-08T20:34:30+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-09-15T15:45:00+02:00
# @License: MIT

# standard types
import networkx as nx
import numpy as np
import pandas as pd
import sys

# custom embeddings/architectures
import kirchhoff.init_crystal as init_crystal
import kirchhoff.init_random as init_random
# custom output functions
import kirchhoff.draw_networkx as dx

def initialize_circuit_from_networkx(input_graph):

    kirchhoff_graph=circuit()
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

def initialize_circuit_from_crystal(crystal_type='default',periods=1):

    kirchhoff_graph=circuit()
    input_graph=init_crystal.init_graph_from_crystal(crystal_type,periods)
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

def initialize_circuit_from_random(random_type='default',periods=10,sidelength=1):

    kirchhoff_graph=circuit()
    input_graph=init_random.init_graph_from_random(random_type,periods,sidelength)
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

class circuit:

    def __init__(self):

        self.scales={
            'conductance':1,
            'flow':1,
            'length':1
        }

        self.graph={
            'source_mode':'',
            'plexus_mode':'',
            'threshold':0.001,
            'num_sources':1
        }

        self.nodes=pd.DataFrame(
        {
            'source':[],
            'potential':[],
            'label':[],
        }
        )

        self.edges=pd.DataFrame(
        {
            'conductivity':[],
            'flow_rate':[],
            'label':[],
        }
        )
        self.set_graph_containers()

        self.draw_weight_scaling=1.

    def set_graph_containers(self):

        self.G=nx.DiGraph()
        self.H=nx.Graph()
        self.H_C=[]
        self.H_J=[]

        self.list_graph_nodes=[]
        self.list_graph_edges=[]

    def default_init(self, input_graph):

        self.G=nx.convert_node_labels_to_integers(input_graph, first_label=0, ordering='default')
        self.initialize_circuit()

        self.list_graph_nodes=list(self.G.nodes())
        self.list_graph_edges=list(self.G.edges())

    def initialize_circuit(self):

        e=self.G.number_of_edges()
        n=self.G.number_of_nodes()

        init_val=['#269ab3',0,0,5]
        init_attributes=['color','source','potential','conductivity']

        for i,val in enumerate(init_val):
            nx.set_node_attributes(self.G, val , name=init_attributes[i])

        for k in self.nodes:
            self.nodes[k]=np.zeros(n)

        for k in self.edges:
            self.edges[k]=np.zeros(e)

        self.set_network_attributes()
        print('circuit(): initialized and ready for (some) action :)')

    #get incidence atrix and its transpose
    def get_incidence_matrices(self):

        B=nx.incidence_matrix(self.G,nodelist=self.list_graph_nodes,edgelist=self.list_graph_edges,oriented=True).toarray()
        BT=np.transpose(B)

        return B,BT

    # update network traits from dynamic data
    def set_network_attributes(self):

        #set potential node values
        for i,n in enumerate(self.list_graph_nodes):

            self.G.nodes[n]['potential']=self.nodes['potential'][i]
            self.G.nodes[n]['label']=i
        #set conductivity matrix
        for j,e in enumerate(self.list_graph_edges):
            self.G.edges[e]['conductivity']=self.edges['conductivity'][j]
            self.G.edges[e]['label']=j

    # clipp small edges & translate conductance into general edge weight
    def clipp_graph(self):

        #cut out edges which lie beneath a certain threshold value and export this clipped structure
        self.set_network_attributes()

        for e in self.list_graph_edges:
            if self.G.edges[e]['conductivity'] > self.threshold:
                self.H.add_edge(*e)
                for k in self.G.edges[e].keys():
                    self.H.edges[e][k]=self.G.edges[e][k]

        self.list_pruned_nodes=list(self.H.nodes())
        self.list_pruned_edges=list(self.H.edges())

        for n in list_pruned_nodes:
            for k in self.G.nodes[n].keys():
                self.H.nodes[n][k]=self.G.nodes[n][k]
            self.H_J.append(self.G.nodes[n]['source'])
        for e in list_pruned_edges:
            self.H_C.append(self.H.edges[e]['conductivity'])

        self.H_C=np.asarray(self.H_C)
        self.H_J=np.asarray(self.H_J)
        if len(list(self.H.nodes()))==0:
            sys.exit('FAILED PRUNING')

    def calc_root_incidence(self):

        root=0
        sink=0

        for i,n in enumerate(self.list_graph_nodes):
            if self.G.nodes[n]['source'] >  0:
                root=n
            if K.G.nodes[n]['source'] <  0:
                sink=n

        E_1=list(self.G.edges(root))
        E_2=list(self.G.edges(sink))
        E_ROOT=[]
        E_SINK=[]
        for e in E_1:
            if e[0]!=root:
                E_ROOT+=list(self.G.edges(e[0]))
            else:
                E_ROOT+=list(self.G.edges(e[1]))

        for e in E_2:
            if e[0]!=sink:
                E_SINK+=list(self.G.edges(e[0]))
            else:
                E_SINK+=list(self.edges(e[1]))

        return E_ROOT,E_SINK

    # test consistency of conductancies & sources
    def test_source_consistency(self):

        self.set_network_attributes()
        tolerance=0.000001
        # check value consistency


        sources=np.fromiter(nx.get_node_attributes(self.G, 'source').values(),float)
        if np.sum(sources) > tolerance:
            sys.exit('Error, input and ouput flows not balanced!')
        else:
            print('set_source_landscape(): '+self.graph['source_mode']+' is set and consistent :)')

    def test_conductance_consistency(self):

        self.set_network_attributes()
        tolerance=0.000001
        # check value consistency

        conductivities=np.fromiter(nx.get_edge_attributes(self.G, 'conductivity').values(),float)
        if len(np.where(conductivities <=0 )[0]) !=0:
            sys.exit('Error, conductivities negaitve/zero!')
        else:
            print('set_plexus_landscape(): '+self.graph['plexus_mode']+' is set and consistent :)')

    def get_pos(self):

        pos_key='pos'
        reset_layout=False
        for j,n in enumerate(self.G.nodes()):
            if pos_key not in self.G.nodes[n]:
                reset_layout=True
        if reset_layout:
            print('set networkx.spring_layout()')
            pos = nx.spring_layout(self.G)
        else:
            pos = nx.get_node_attributes(self.G,'pos')

        return pos

    def set_pos(self,pos_data={}):

        pos_key='pos'
        reset_layout=False
        nodata=False
        if len(pos_data.values())==0:
            nodata=True

        for j,n in enumerate(self.G.nodes()):
            if pos_key not in self.G.nodes[n]:
                reset_layout=True
        if reset_layout and nodata:
            print('set networkx.spring_layout()')
            pos = nx.spring_layout(self.G)
            nx.set_node_attributes(self.G, pos, 'pos')
        else:
            nx.set_node_attributes(self.G, pos_data, 'pos')


    def set_scale_pars(self, new_parameters):

        self.scales=new_parameters

    def set_graph_pars(self, new_parameters):

        self.graph=new_parameters

    # output
    def plot_circuit(self,*args,**kwargs):

        self.set_pos()

        E=self.get_edges_data(*args)
        V=self.get_nodes_data(*args)

        fig=dx.plot_networkx(   self.G, edge_list=self.list_graph_edges, node_list=self.list_graph_nodes, edge_data=E,  node_data=V ,**kwargs)
        return fig

    def get_nodes_data(self, *args ):

        # dn=pd.DataFrame(self.nodes[['label']])
        cols=['label']
        cols+=[a for a in args if a in self.nodes.columns]

        dn=pd.DataFrame(self.nodes[cols])

        return dn

    def get_edges_data(self, *args ):

        cols=['label']
        cols+=[a for a in args if a in self.edges.columns]

        de=pd.DataFrame(self.edges[cols])

        return de
