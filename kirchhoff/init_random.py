# @Author:  Felix Kramer
# @Date:   2021-05-22T13:11:37+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-05-22T22:15:12+02:00
# @License: MIT

import networkx as nx
import scipy as sc
from scipy.spatial import Voronoi
import random as rd

def init_graph_from_random(random_type,periods,sidelength):

    choose_constructor_option={ 'default': networkx_voronoi_planar, 'voronoi_planar': networkx_voronoi_planar, 'voronoi_volume': networkx_voronoi_volume}

    if random_type in choose_constructor_option:
            random=choose_constructor_option[random_type](periods,sidelength)

    else :
        print('Warning, crystal type unknown, set default: simple')
        random=choose_constructor_option['default'](periods,sidelength)

    return random.G

class networkx_random():

    def __init__(self):

        self.G=nx.Graph()

    def mirror_boxpoints(self,points,sl):

        points_matrix=points
        intervall=[-1,0,1]
        for i in intervall:
            for j in intervall:
                if (i!=0 or j!=0):
                    points_matrix=sc.concatenate((points_matrix,points+(i*sl,j*sl)))

        return points_matrix

    def mirror_cubepoints(self,points,sl):

        points_matrix=points
        intervall=[-1,0,1]
        for i in intervall:
            for j in intervall:
                for k in intervall:
                    if (i!=0 or j!=0 or k!=0):
                        points_matrix=sc.concatenate((points_matrix,points+(i*sl,j*sl,k*sl)))

        return points_matrix
    # construct random 3d graph, confined in a box

    def is_in_box(self,v,sl):
        answer=True

        if (v[0] > sl) or (v[0] < -sl):
            answer=False
        if (v[1] > sl) or (v[1] < -sl):
            answer=False
        if (v[2] > sl) or (v[2] < -sl):
            answer=False

        return answer

class networkx_voronoi_planar(networkx_random,object):

    def __init__(self, num_periods,sidelength):

        super(networkx_voronoi_planar,self).__init__()
        self.random_voronoi_periodic(num_periods,sidelength)
        # construct random 2d graph, confined in a certain spherical boundary, connections set via voronoi tesselation

    def construct_voronoi_periodic(self,number,sidelength):

        V=0
        # create points for voronoi tesselation
        XY=[]

        for i in range(number):
            x=rd.uniform(0,sidelength)
            y=rd.uniform(0,sidelength)

            XY.append((x,y))
        self.XY=XY
        XY=self.mirror_boxpoints(sc.array(XY),sidelength)
        self.XY_periodic=XY
        V=Voronoi(XY)

        return V

    def random_voronoi_periodic(self,number,sidelength):

        #construct a core of reandom points in 2D box for voronoi tesselation, mirror the primary box so a total of 9 cells is created with the initial as core
        V=self.construct_voronoi_periodic(number,sidelength)
        #pick up the face of the core which correspond to a periodic voronoi lattice
        faces=[]
        for j,i in enumerate(V.point_region):
            faces.append(sc.asarray(V.regions[i]))

            if j==number-1:
                break
        #use periodic kernel to construct the correponding network
        faces=sc.asarray(faces)
        f=faces[0]


        for i in range(len(faces[:])):
            if i+1==len(faces[:]):
                break
            f=sc.concatenate((f,faces[i+1]))
        for i in faces:
            for j in i:
                v=V.vertices[j]
                self.G.add_node(j,pos=v,lablel=j)

        k=0
        for i in V.ridge_vertices:

            mask=sc.in1d(i,f)
            if sc.all( mask == True ):

                for l in range(len(i)):
                    h=len(i)-1
                    self.G.add_edge(i[h-(l+1)],i[h-l],slope=(V.vertices[i[h-(l+1)]],V.vertices[i[h-l]]), label=k)
                    k+=1
                    if len(i)==2:
                        break

class networkx_voronoi_volume(networkx_random,object):

    def __init__(self, num_periods,sidelength):

        super(networkx_voronoi_volume,self).__init__()
        self.random_voronoi_periodic(num_periods,sidelength)
    # construct random 3d graph, confined in a certain spherical boundary, connections set via voronoi tesselation

    def construct_voronoi_periodic(self,number,sidelength):

        V=0
        # create points for voronoi tesselation

        XYZ=[]

        for i in range(number):
            x=rd.uniform(0,sidelength)
            y=rd.uniform(0,sidelength)
            z=rd.uniform(0,sidelength)

            XYZ.append((x,y,z))
        self.XYZ=XYZ
        XYZ=self.mirror_cubepoints(sc.array(XYZ),sidelength)
        self.XYZ_periodic=XYZ
        V=Voronoi(XYZ)

        return V

    def random_voronoi_periodic(self,number,sidelength):

        #construct a core of reandom points in 2D box for voronoi tesselation, mirror the primary box so a total of 9 cells is created with the initial as core
        V=self.construct_voronoi_periodic(number,sidelength)
        #pick up the face of the core which correspond to a periodic voronoi lattice
        faces=[]
        for j,i in enumerate(V.point_region):
            faces.append(sc.asarray(V.regions[i]))

            if j==number-1:
                break
        #use periodic kernel to construct the correponding network
        faces=sc.asarray(faces)
        f=faces[0]


        for i in range(len(faces[:])):
            if i+1==len(faces[:]):
                break
            f=sc.concatenate((f,faces[i+1]))
        for i in faces:
            for j in i:
                v=V.vertices[j]
                self.G.add_node(j,pos=v,lablel=j)

        k=0
        for i in V.ridge_vertices:

            mask=sc.in1d(i,f)
            if sc.all( mask == True ):

                for l in range(len(i)):
                    h=len(i)-1
                    self.G.add_edge(i[h-(l+1)],i[h-l],slope=(V.vertices[i[h-(l+1)]],V.vertices[i[h-l]]), label=k)
                    k+=1
                    if len(i)==2:
                        break
