# @Author:  Felix Kramer
# @Date:   2021-05-22T13:11:37+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-07T12:15:36+01:00
# @License: MIT
import networkx as nx
import numpy as np

# construct a non-trivial,  periodic 3d embedding
def init_graph_from_crystal(crystal_type, periods):

    choose_constructor_option = {
        'default': networkx_simple,
        'simple': networkx_simple,
        'chain': networkx_chain,
        'bcc': networkx_bcc,
        'fcc': networkx_fcc,
        'diamond': networkx_diamond,
        'laves': networkx_laves,
        'square': networkx_square,
        'hexagonal': networkx_hexagonal,
        'trigonal_planar': networkx_trigonal_planar,
        }

    if crystal_type in choose_constructor_option:
            crystal = choose_constructor_option[crystal_type](periods)

    else :
        print('Warning, crystal type unknown, set default: simple')
        crystal = choose_constructor_option['default'](1)

    return crystal.G

def init_graph_from_asymCrystal(crystal_type, periodsZ, periodsXY):

    choose_constructor_option = {
        'trigonal_stack': networkx_trigonal_stack,
        }

    if crystal_type in choose_constructor_option:
            crystal = choose_constructor_option[crystal_type](periodsZ, periodsXY)

    else :
        print('Warning, crystal type unknown, set default: trigonal_stack')
        crystal = choose_constructor_option['default'](2,2)

    return crystal.G


class networkx_crystal():

    def __init__(self):
        self.dict_cells = {  }
        self.G = nx.Graph()
        self.lattice_constant = 1
        self.translation_length = 1

    # construct one of the following crystal topologies
    def lattice_translation(self, t, T):

        D = nx.Graph()
        for n in T.nodes():
            D.add_node(tuple(n+t), pos=T.nodes[n]['pos']+t)

        return D

    def periodic_cell_structure(self, cell, num_periods):

        DL = nx.Graph()

        if type(num_periods) is not int :
            periods = [range(num_periods[i]) for i in range(3)]
        else:
            periods = [range(num_periods) for i in range(3)]
        for i in periods[0]:
            for j in periods[1]:
                for k in periods[2]:
                    v = self.translation_length*np.array([i, j, k])
                    TD = self.lattice_translation(v, cell)
                    DL.add_nodes_from(TD.nodes(data=True))
                    self.dict_cells[(i, j, k)] = list(TD.nodes())

        list_n = np.array(list(DL.nodes()))
        for i, n in enumerate(list_n[:-1]):
            for m in list_n[(i+1):]:
                p = DL.nodes[tuple(n)]['pos']
                q = DL.nodes[tuple(m)]['pos']
                dist = np.linalg.norm(p-q)
                if dist == self.lattice_constant:
                    mn = (DL.nodes[tuple(n)]['pos'], DL.nodes[tuple(m)]['pos'])
                    DL.add_edge(tuple(n), tuple(m), slope=mn)

        dict_nodes = {}
        for idx_n, n in enumerate(DL.nodes()):
            self.G.add_node(idx_n, pos=DL.nodes[n]['pos'])
            dict_nodes.update({n:idx_n})
        for idx_e, e in enumerate(DL.edges()):
            u, v = dict_nodes[e[0]], dict_nodes[e[1]]
            uv = (DL.nodes[e[0]]['pos'], DL.nodes[e[1]]['pos'])
            self.G.add_edge(u, v, slope=uv )

        self.dict_cubes = {}
        dict_aux = {}
        for i, k in enumerate(self.dict_cells.keys()):
            dict_aux[i] = [dict_nodes[n] for n in self.dict_cells[k]]
        for i, k in enumerate(dict_aux.keys()):
            self.dict_cubes[k] = nx.Graph()
            n_list = list(dict_aux[k])
            for u in n_list[:-1]:
                for v in n_list[1:]:
                    if self.G.has_edge(u, v):
                        self.dict_cubes[k].add_edge(u, v)
# 3D
class networkx_simple(networkx_crystal, object):

    def __init__(self,  num_periods):

        super(networkx_simple, self).__init__()
        self.lattice_constant = 1.
        self.translation_length = 1.
        self.simple_cubic_lattice(num_periods)

    #construct full cubic grid as skeleton
    def simple_unit_cell(self):

        D = nx.Graph()
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    D.add_node(tuple((i, j, k)), pos=np.array([i, j, k]))

        return D

    def simple_cubic_lattice(self, num_periods):

        D = self.simple_unit_cell()
        self.periodic_cell_structure(D, num_periods)

class networkx_chain(networkx_crystal, object):

    def __init__(self, num_periods):

        super(networkx_chain, self).__init__()
        self.simple_chain(num_periods)

    def simple_chain(self, num_periods):

        #construct single box
        for i in range(num_periods):
            self.G.add_node(i, pos=np.array([i, 0, 0]))
        for i in range(num_periods-1):
            df = (self.G.nodes[i+1]['pos'], self.G.nodes[i]['pos'])
            self.G.add_edge(i+1, i, slope=df)

class networkx_bcc(networkx_crystal, object):

    def __init__(self,  num_periods):
        super(networkx_bcc, self).__init__()
        self.lattice_constant = np.sqrt(3.)/2.
        self.translation_length = 1.
        self.simple_bcc_lattice(num_periods)

    def bcc_unit_cell(self):

        D = nx.Graph()
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    D.add_node(tuple((i, j, k)), pos = np.array([i, j, k]))
        D.add_node(tuple((0.5, 0.5, 0.5)), pos=np.array([0.5, 0.5, 0.5]))

        return D

    def simple_bcc_lattice(self, n):

        #construct single box
        D = self.bcc_unit_cell()
        self.periodic_cell_structure(D, n)

class networkx_fcc(networkx_crystal, object):

    def __init__(self, num_periods):
        super(networkx_fcc, self).__init__()
        self.lattice_constant = np.sqrt(2.)/2.
        self.translation_length = 1.
        self.simple_fcc_lattice(num_periods)

    def fcc_unit_cell(self):

        D = nx.Graph()
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    D.add_node(tuple((i, j, k)), pos=np.array([i, j, k]))
        for i in [0., 1.]:
            D.add_node(tuple((0.5, i, 0.5)), pos=np.array([0.5, i, 0.5]))
        for i in [0., 1.]:
            D.add_node(tuple((0.5, 0.5, i)), pos=np.array([0.5, 0.5, i]))
        for i in [0., 1.]:
            D.add_node(tuple((i, 0.5, 0.5)), pos=np.array([i, 0.5, 0.5]))

        return D

    def simple_fcc_lattice(self, n):

        D = self.fcc_unit_cell()
        self.periodic_cell_structure(D, n)

class networkx_diamond(networkx_crystal, object):

    def __init__(self, num_periods):
        super(networkx_diamond, self).__init__()
        self.lattice_constant = np.sqrt(3.)/2.
        self.translation_length = 2.
        self.diamond_lattice(num_periods)

    def diamond_unit_cell(self):

        D = nx.Graph()
        T = [nx.Graph() for i in range(4)]
        T[0].add_node((0, 0, 0), pos=np.array([0, 0, 0]))
        T[0].add_node((1, 1, 0), pos=np.array([1, 1, 0]))
        T[0].add_node((1, 0, 1), pos=np.array([1, 0, 1]))
        T[0].add_node((0, 1, 1), pos=np.array([0, 1, 1]))
        T[0].add_node((0.5, 0.5, 0.5), pos=np.array([0.5, 0.5, 0.5]))
        translation = [np.array([1, 1, 0]), np.array([1, 0, 1]),
         np.array([0, 1, 1])]
        for i, t in enumerate(translation):
            for n in T[0].nodes():
                T[i+1].add_node(tuple(n+t), pos=T[0].nodes[n]['pos']+t)
        for t in T:
            D.add_nodes_from(t.nodes(data=True))

        return D

    def diamond_lattice(self, num_periods):

        D = self.diamond_unit_cell()
        self.periodic_cell_structure(D, num_periods)

class networkx_laves(networkx_crystal, object):

    def __init__(self, num_periods):
        super(networkx_laves, self).__init__()
        self.lattice_constant = 2.
        self.laves_lattice(num_periods)

    def laves_lattice(self, num_periods):

        #construct single box
        counter = 0
        G_aux = nx.Graph()
        if type(num_periods) is not int :
            periods = [range(num_periods[i]) for i in range(3)]
        else:
            periods = [range(num_periods) for i in range(3)]

        fundamental_points = [[0, 0, 0], [1, 1, 0], [1, 2, 1], [0, 3, 1],
         [2, 2, 2], [3, 3, 2], [3, 0, 3], [2, 1, 3]]
        for l, fp in enumerate(fundamental_points):
            for i in periods[0]:
                for j in periods[1]:
                    for k in periods[2]:

                        pos_n = np.add(fp, [4.*i, 4.*j, 4.*k])
                        G_aux.add_node(tuple(pos_n), pos=pos_n)

        list_nodes = list(G_aux.nodes())
        self.G = nx.Graph()
        H = nx.Graph()
        points_G = [G_aux.nodes[n]['pos'] for i, n in enumerate(G_aux.nodes()) ]
        for i, n in enumerate(G_aux.nodes()) :
            H.add_node(n, pos=G_aux.nodes[n]['pos'])
        for i, n in enumerate(list_nodes[:-1]):
              for j, m in enumerate(list_nodes[(i+1):]):

                v = np.subtract(n, m)
                dist = np.dot(v, v)
                if dist == self.lattice_constant:
                    mn = (G_aux.nodes[n]['pos'], G_aux.nodes[m]['pos'])
                    H.add_edge(n, m, slope=mn)

        dict_nodes = {}
        for idx_n, n in enumerate(H.nodes()):
            self.G.add_node(idx_n, pos=H.nodes[n]['pos'])
            dict_nodes.update({n:idx_n})

        for idx_e, e in enumerate(H.edges()):
            u,v = dict_nodes[e[0]], dict_nodes[e[1]]
            uv = (H.nodes[e[0]]['pos'], H.nodes[e[1]]['pos'])
            self.G.add_edge(u,v, slope=uv)

class networkx_trigonal_stack(networkx_crystal, object):

    def __init__(self, stacks, tiling_factor):
        super(networkx_trigonal_stack, self).__init__()
        self.triangulated_hexagon_stack(stacks, tiling_factor)

    #define crosslinking procedure between the generated single-layers
    def crosslink_stacks(self):

      for i, n in enumerate(self.G.nodes()):
          self.G.nodes[n]['label'] = i
      if self.stacks > 1 :
          labels_n  =  nx.get_node_attributes(self.G, 'label')
          sorted_label_n_list = sorted(labels_n , key = labels_n.__getitem__)
          # for n in nx.nodes(self.G):
          for n in sorted_label_n_list:
              if n[2]!= self.stacks-1:

                  u = (n[0], n[1], n[2])
                  v = (n[0], n[1], n[2]+1)
                  p1 = self.G.nodes[u]['pos']
                  p2 = self.G.nodes[v]['pos']

                  self.G.add_edge(u, v, slope=(p1, p2))

    # auxillary function,  construct triangulated hex grid upper and lower wings
    def construct_spine_stack(self, z, n):

        self.spine = 2*(n-1)
        self.G.add_node((0, 0, z), pos=(0., 0., z))
        for m in range(self.spine):

            self.G.add_node((m+1, 0, z), pos=((m+1), 0., z))
            u = (m, 0, z)
            v = (m+1, 0, z)
            uv = (self.G.nodes[u]['pos'], self.G.nodes[v]['pos'])

            self.G.add_edge(u, v, slope=uv)

    def construct_wing_stack(self, z, a, n):

        for m in range(n-1):
            #m-th floor
            floor_m_nodes = self.spine-(m+1)

            u = (0, a*(m+1), z)
            v = (0, a*m, z)
            w = (1, a*m, z)

            self.G.add_node(u, pos = ((m+1)/2., a*(np.sqrt(3.)/2.)*(m+1), z))
            uv = (self.G.nodes[u]['pos'], self.G.nodes[v]['pos'])
            vw = (self.G.nodes[u]['pos'], self.G.nodes[w]['pos'])
            self.G.add_edge(u, v, slope=uv)
            self.G.add_edge(u, w, slope=vw)

            for p in range(floor_m_nodes):
                #add 3-junctions
                u = (p+1, a*(m+1), z)
                v = (p+1, a*m, z)
                w = (p+2, a*m, z)
                x = (p, a*(m+1), z)

                ps = (((p+1)+(m+1)/2.), a*(np.sqrt(3.)/2.)*(m+1), z)
                self.G.add_node(u, pos = ps)

                uv = (self.G.nodes[u]['pos'], self.G.nodes[v]['pos'])
                vw = (self.G.nodes[v]['pos'], self.G.nodes[w]['pos'])
                ux = (self.G.nodes[u]['pos'], self.G.nodes[x]['pos'])

                self.G.add_edge(u, v, slope=uv)
                self.G.add_edge(u, w, slope=vw)
                self.G.add_edge(u, x, slope=ux)

    #construct full triangulated hex grids as skeleton of a stacked structure
    def triangulated_hexagon_stack(self, stack, n):

      self.stacks = stack
      for z in range(self.stacks):

          #construct spine for different levels of lobule
          self.construct_spine_stack(z, n)

          #construct lower/upper halfspace
          self.construct_wing_stack(z, -1,  n)
          self.construct_wing_stack(z,  1,  n)

      self.crosslink_stacks()

# 2D
class networkx_square(networkx_crystal, object):

    def __init__(self, tiling_factor):
        super(networkx_square, self).__init__()
        self.square_grid(tiling_factor)

    def square_grid(self, num_periods):

        if type(num_periods) is not int :
            a = [range(0, num_periods[0]+1), range(0, num_periods[1]+1)]
        else:
            a = [range(0, num_periods+1), range(0, num_periods+1)]

        for x in a[0]:
            for y in a[1]:
                self.G.add_node((x, y), pos=(x, y, 0))

        list_n = list(self.G.nodes())
        dict_d = {}
        threshold = 1.
        for idx_n, n in enumerate(list_n[:-1]):
            for m in list_n[idx_n+1:]:
                p = np.array(self.G.nodes[n]['pos'])
                q = np.array(self.G.nodes[m]['pos'])
                dict_d[(n, m)] = np.linalg.norm(p-q)

        for nm in dict_d:
            if dict_d[nm] <=  threshold:
                dnm = [self.G.nodes[nm[0]]['pos'], self.G.nodes[nm[1]]['pos']]
                self.G.add_edge(*nm, slope=dnm)

class networkx_trigonal_planar(networkx_crystal, object):

    def __init__(self,  tiling_factor):
        super(networkx_trigonal_planar, self).__init__()
        self.triangulated_hexagon_lattice(tiling_factor)
    #I) construct and define one-layer hex
    # auxillary function,  construct triangulated hex grid upper and lower wings
    def construct_wing(self, a, n):

      for m in range(n-1):
        #m-th floor
        floor_m_nodes = self.spine - (m+1)

        u = (0, a*(m+1))
        v = (0, a*m)
        w = (1, a*m)
        ps = np.array([(m+1)/2., a*(np.sqrt(3.)/2.)*(m+1)])

        self.G.add_node(u, pos=ps)
        uv = (self.G.nodes[u]['pos'], self.G.nodes[v]['pos'])
        uw = (self.G.nodes[u]['pos'], self.G.nodes[w]['pos'])
        self.G.add_edge(u, v, slope=uv)
        self.G.add_edge(u, w, slope=uw)

        for p in range(floor_m_nodes):
            #add 3-junctions
            u = (p+1, a*(m+1))
            v = (p+1, a*m)
            w = (p+2, a*m)
            x = (p, a*(m+1))
            ps = np.array([((p+1)+(m+1)/2.), a*(np.sqrt(3.)/2.)*(m+1)])

            self.G.add_node(u, pos=ps)

            uv = (self.G.nodes[u]['pos'], self.G.nodes[v]['pos'])
            uw = (self.G.nodes[u]['pos'], self.G.nodes[w]['pos'])
            ux = (self.G.nodes[u]['pos'], self.G.nodes[x]['pos'])
            self.G.add_edge(u, v, slope=uv)
            self.G.add_edge(u, w, slope=uw)
            self.G.add_edge(u, x, slope=ux)

    #construct full triangulated hex grid as skeleton
    def triangulated_hexagon_lattice(self, n):

      #construct spine
      self.spine = 2*(n-1)
      self.G.add_node((0, 0), pos=np.array([0., 0.]))

      for m in range(self.spine):

          self.G.add_node((m+1, 0), pos=np.array([(m+1), 0.]))
          u, v = (m, 0), (m+1, 0)
          uv = (self.G.nodes[u]['pos'], self.G.nodes[v]['pos'])
          self.G.add_edge( u, v, slope=uv)

      #construct lower/upper halfspace
      self.construct_wing(-1, n)
      self.construct_wing(1, n)

class networkx_hexagonal(networkx_crystal, object):

        def __init__(self, tiling_factor, periodic=False):
            super(networkx_hexagonal, self).__init__()
            self.hexagonal_grid(tiling_factor, periodic)

        def hexagonal_grid(self, tiling_factor, periodic_bool):

            m = 2*tiling_factor+1
            n = 2*tiling_factor
            opt = {
                'periodic': periodic_bool,
                'with_positions': True
            }
            self.G = nx.hexagonal_lattice_graph(m, n, **opt)

            for n in self.G.nodes():
                self.G.nodes[n]['pos'] = np.array(self.G.nodes[n]['pos'])
            for e in self.G.edges():
                slope =  [self.G.nodes[e[0]]['pos'], self.G.nodes[e[1]]['pos']]
                self.G.edges[e]['slope'] = slope
