# @Author:  Felix Kramer <kramer>
# @Date:   2021-05-08T20:35:25+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-07T16:06:39+01:00
# @License: MIT

import random as rd
import networkx as nx
import numpy as np
import sys
import pandas as pd
# custom embeddings/architectures
from kirchhoff.circuit_init import *
import kirchhoff.init_crystal as init_crystal
import kirchhoff.init_random as init_random


def initialize_flow_circuit_from_networkx(input_graph):

    """
    Initialize a flow circuit from a given networkx graph.

    Args:
        input_graph (nx.Graph): A networkx graph.

    Returns:
        flow_circuit: A flow_circuit object.

    """

    kirchhoff_graph = flow_circuit()
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

def initialize_flow_circuit_from_crystal(crystal_type='default', periods=1):

    """
    Initialize a flow circuit from a spatially embedded, crystal networkx graph.

    Args:
        crystal_type (string): The type of crystal skeleton (default, simple, chain, bcc, fcc, diamond, laves, square, hexagonal, trigonal_planar).
        periods (int): Repetition number of the lattice's unit cell.

    Returns:
        flow_circuit: A flow_circuit object.

    """

    kirchhoff_graph = flow_circuit()
    input_graph = init_crystal.init_graph_from_crystal(crystal_type, periods)
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

def initialize_flow_circuit_from_random(random_type='default', periods=10, sidelength=1):

    """
    Initialize a flow circuit from a random, spatially embedded networkx graph.

    Args:
        random_type (string): The type of random lattice to be constructed(voronoi_planar, voronoi_volume).
        periods (int): Number of random points.
        sidelength (float): The box size into which random points in space are generated.

    Returns:
        flow_circuit: A flow_circuit object.

    """

    kirchhoff_graph = flow_circuit()
    input_graph = init_random.init_graph_from_random(random_type, periods, sidelength)
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

def setup_default_flow_circuit(skeleton=None):

    """
    Initialize a flow circuit from a given networkx graph.

    Args:
        skeleton (nx.Graph): A networkx graph.

    Returns:
        flow_circuit: A flow_circuit object.

    """

    kirchhoff_graph = initialize_flow_circuit_from_networkx(skeleton)
    kirchhoff_graph.set_source_landscape()
    kirchhoff_graph.set_plexus_landscape()

    return kirchhoff_graph

def setup_flow_circuit(skeleton=None, source=None, plexus=None):

    """
    Initialize a flow circuit from a given networkx graph and dictioinary of boundary conditions.

    Args:
        skeleton (nx.Graph): A networkx graph.
        source (string): A key for source_mode (default, root_geometric, root_short, root_long, dipole_border, dipole_point, root_multi, custom).
        plexus (string): A key for plexus_mode.(default, custom)

    Returns:
        flow_circuit: A flow_circuit object.

    """

    kirchhoff_graph = initialize_flow_circuit_from_networkx(skeleton)
    kirchhoff_graph.set_source_landscape(source)
    kirchhoff_graph.set_plexus_landscape(plexus)

    return kirchhoff_graph
# class flow_circuit(kirchhoff_init.circuit, object):
class flow_circuit(circuit, object):

    """
    A derived class for flow circuits.

    Attributes
    ----------

        source_mode (dictionary): A dictionary of custom source-sink boundaries.
        plexus_mode (dictionary): A dictionary of custom plexus initializations.

    """

    def __init__(self):

        """
        A derived constructor for flow circuits, setting source_mode and plexus_mode.
        """

        super(flow_circuit, self).__init__()

        self.source_mode = {
            'default': self.init_source_default,
            'root_geometric': self.init_source_root_central_geometric,
            'root_short': self.init_source_root_short,
            'root_long': self.init_source_root_long,
            'dipole_border': self.init_source_dipole_border,
            'dipole_point': self.init_source_dipole_point,
            'root_multi': self.init_source_root_multi,
            'custom': self.init_source_custom
        }

        self.plexus_mode = {
            'default': self.init_plexus_default,
            'custom': self.init_plexus_custom,

        }

    # set a certain set of boundary conditions for the given networks
    def set_source_landscape(self, mode='default', **kwargs):

        """
        Set the internal bounday state of sinks and sources.

        Args:
            mode (string): The specific source mode.
            kwargs (dictonary): Source attribute specifiers, optional.

        """

        # optional keywords
        if 'num_sources' in kwargs:
            self.graph['num_sources'] = kwargs['num_sources']

        elif 'sources' in kwargs:
            self.custom = kwargs['sources']

        # else:
        #     print('Warning: Not recognizing certain keywords')
        # call init sources
        if mode in self.source_mode.keys():

            self.source_mode[mode]()

        else :
            sys.exit('Whooops,  Error: Define Input/output-flows for the network.')

        self.graph['graph_mode'] = mode
        self.test_source_consistency()

    def set_potential_landscape(self, mode):

        # todo
        return 0

    # different init source functions
    def init_source_custom(self):

        """
        Set custom sinks and sources boundaries.
        """

        if len(self.custom.keys()) == len(self.list_graph_nodes):

            for j, node in enumerate(self.list_graph_nodes):

                s = self.custom[node]*self.scales['flow']
                self.G.nodes[node]['source'] = s
                self.nodes['source'][j] = s

        else:
            print('Warning, custom source values ill defined, setting default!')
            self.init_source_default()

    def init_source_default(self):

        """
        Set one topologically central source, sinks otherwise.
        """

        centrality = nx.betweenness_centrality(self.G)
        centrality_sorted = sorted(centrality, key=centrality.__getitem__)

        self.set_root_leaves_relationship(centrality_sorted[-1])

    def init_source_root_central_geometric(self):

        """
        Set one geometrically central source, sinks otherwise.
        """

        pos = self.get_pos()
        X = np.mean(list(pos.values()), axis=0)

        dist = {}
        for n in self.list_graph_nodes:
            dist[n] = np.linalg.norm(np.subtract(X, pos[n]))
        sorted_dist = sorted(dist, key=dist.__getitem__)

        self.set_root_leaves_relationship(sorted_dist[0])

    def init_source_root_short(self):

        """
        Set one-sided source (right), sinks otherwise.
        """

        # check whether geometric layout has been set
        pos = self.get_pos()

        # check for root closests to coordinate origin
        dist = {}
        for n, p in pos.items():
            dist[n] = np.linalg.norm(p)
        sorted_dist = sorted(dist, key = dist.__getitem__)

        self.set_root_leaves_relationship(sorted_dist[0])

    def init_source_root_long(self):

        """
        Set one-sided source (left), sinks otherwise.
        """

        # check whether geometric layout has been set
        pos = self.get_pos()

        # check for root closests to coordinate origin
        dist = {}
        for n, p in pos.items():
            dist[n] = np.linalg.norm(p)
        sorted_dist = sorted(dist, key=dist.__getitem__, reverse=True)

        self.set_root_leaves_relationship(sorted_dist[0])

    def init_source_dipole_border(self):

        """
        Set sources on one side of the graph, sinks on the opposing side.
        """

        pos = self.get_pos()
        dist = {}
        for n, p in pos.items():
            dist[n] = np.linalg.norm(p[0])

        vals = list(dist.values())
        max_x = np.amax(vals)
        min_x = np.amin(vals)

        max_idx = []
        min_idx = []
        for k, v in dist.items():
            if v == max_x:
                max_idx.append(k)

            elif v == min_x:
                min_idx.append(k)

        self.set_poles_relationship(max_idx, min_idx)

    def init_source_dipole_point(self):

        """
        Set a single source-sink pair.
        """

        pos = self.get_pos()
        dist = {}
        for j, n in enumerate(self.list_graph_nodes[:-2]):
            for i, m in enumerate(self.list_graph_nodes[j+1:]):
                path = nx.shortest_path(self.G, source=n, target=m)
                dist[(n, m)] = len(path)
        max_len = np.amax(list(dist.values()))
        push = []
        for key in dist.keys():
            if dist[key] == max_len:
                push.append(key)

        idx = np.random.choice(range(len(push)))
        source, sink = push[idx]

        self.set_poles_relationship([source], [sink])

    def init_source_root_multi(self):

        """
        Set multiple random sources, sinks otherwise.
        """

        idx = np.random.choice(self.list_graph_nodes, size=self.graph['num_sources'])
        self.nodes_source = [self.G.number_of_nodes()/self.graph['num_sources']-1, -1]

        for j, n in enumerate(self.list_graph_nodes):

            if n in idx:

                self.set_source_attributes(j, n, 0)

            else:

                self.set_source_attributes(j, n, 1)

    # auxillary function for the block above
    def set_root_leaves_relationship(self, root):

        """
        Set boundaries with distinguished root vertex.
        """

        self.nodes_source = [self.G.number_of_nodes()-1, -1]
        for j, n in enumerate(self.list_graph_nodes):

            if n == root:
                idx = 0

            else:
                idx = 1

            self.set_source_attributes(j, n, idx)

    def set_poles_relationship(self, sources, sinks):

        """
        Set boundaries with distinguished pole vertices.
        """

        self.nodes_source = [1, -1, 0]

        for j, n in enumerate(self.list_graph_nodes):
            self.set_source_attributes(j, n, 2)

        for i, s in enumerate(sources):
            for j, n in enumerate(self.list_graph_nodes):
                if n == s:
                    self.set_source_attributes(j, s, 0)

        for i, s in enumerate(sinks):
            for j, n in enumerate(self.list_graph_nodes):
                if n == s:
                    self.set_source_attributes(j, s, 1)

    def set_source_attributes(self, j, node, idx):

        """
        Set nodal flow attributes.
        """

        self.G.nodes[node]['source'] = self.nodes_source[idx]*self.scales['flow']
        self.nodes['source'][j] = self.nodes_source[idx]*self.scales['flow']

    # different init potetnial functions
    def set_terminals_potentials(self, p0):

        """
        Set nodal potentials.
        """

        idx_potential = []
        idx_sources = []
        for j, n in enumerate(nx.nodes(self.G)):

            if self.G.nodes[n]['source'] > 0:
                self.G.nodes[n]['potential'] = 1
                self.V[j] = p0
                idx_potential.append(j)
            elif self.G.nodes[n]['source'] < 0:

                self.G.nodes[n]['potential'] = 0.
                self.V[j] = 0.
                idx_potential.append(j)
            else:
                self.G.nodes[n]['source'] = 0.
                self.J[j] = 0.
                idx_sources.append(j)

        self.G.graph['sources'] = idx_sources
        self.G.graph['potentials'] = idx_potential

    # different init plexus functions
    def set_plexus_landscape(self, mode='default', **kwargs):

        """
        Set the intial conductivity landscape of the plexus.

        Args:
            mode (string): The specific plexus mode.
            kwargs (dictonary): Plexus attribute specifiers, optional.
        """

        # optional keywords

        if 'plexus' in kwargs:
            self.custom = kwargs['plexus']

        # call init sources
        if mode in self.plexus_mode.keys():

            self.plexus_mode[mode]()

        else :
            sys.exit('Whooops,  Error: Define proper conductancies for  the network.')

        self.graph['plexus_mode'] = mode
        self.test_conductance_consistency()

    def init_plexus_default(self):

        """
        Set random initial plexus.
        """

        # find magnitude of flows and set scale of for conductancies
        d = np.amax(self.nodes['source']) * 0.5
        m = self.G.number_of_edges()

        displaced = np.add(np.ones(m), np.random.rand(m))
        self.edges['conductivity'] = np.multiply(d, displaced)

    def init_plexus_custom(self):

        """
        Set customized initial plexus.
        """

        if len(self.custom.keys()) == len(self.list_graph_edges):
            # find magnitude of flows and set scale of for conductancies
            for j, edge in enumerate(self.list_graph_edges):

                c = self.custom[edge]*self.scales['conductance']
                self.G.edges[edge]['conductivity'] = c
                self.edges['conductivity'][j] = c
        else:

            print('Warning,  custom conductance values ill defined,  setting default !')
            self.init_plexus_default()

    # output with draw_netowrkx
    def get_nodes_data(self, *args):

        """
        Get internal nodal DataFrame columns by keywords.

        Args:
            args (list): A list of keywords to check for in the internal DataFrames.

        Returns:
            pd.DataFrame: A cliced DataFrame.

        Raises:
            Exception: description

        """

        cols = ['label', 'source']
        cols+= [a for a in args if a in self.nodes.columns]

        dn = pd.DataFrame(self.nodes[cols])

        return dn

    def get_edges_data(self, *args):

        """
        Get internal nodal DataFrame columns by keywords.

        Args:
            args (list): A list of keywords to check for in the internal DataFrames.

        Returns:
            pd.DataFrame: A cliced DataFrame.

        Raises:
            Exception: description

        """

        cols = ['label', 'conductivity', 'flow_rate']
        cols += [a for a in args if a in self.edges.columns]

        de = pd.DataFrame(self.edges[cols])

        return de

    def plot_circuit(self, *args, **kwargs):

        """
        Use Plotly.GraphObjects to create interactive plots that have
         optionally the graph atributes displayed.
        Args:
            args (list): A list of keywords for the internal edge and nodal DataFrames which are to be displayed.
            kwargs (dictionary): A dictionary for plotly keywords customizing the plots' layout.

        Returns:
            plotly.graph_objects.Figure: A plotly figure displaying the circuit.

        """

        self.set_pos()
        E = self.get_edges_data(*args)
        V = self.get_nodes_data(*args)

        opt = {
            'edge_list': self.list_graph_edges,
            'node_list': self.list_graph_nodes,
            'edge_data': E,
            'node_data': V
        }
        if type(kwargs) != None:
            opt.update(kwargs)
        # print(opt)
        fig = dx.plot_networkx(self.G, **opt)

        return fig
