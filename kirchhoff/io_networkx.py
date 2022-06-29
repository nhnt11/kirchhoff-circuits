# @Author: Felix Kramer <felixk1990>
# @Date:   2022-06-28T23:45:46+02:00
# @Email:  felixuwekramer@proton.me
# @Filename: io_networkx.py
# @Last modified by:   felixk1990
# @Last modified time: 2022-06-28T23:47:03+02:00
import networkx.readwrite.json_graph as nj
import json
import numpy as np
import networkx as nx

def loadGraphJson(pathInput):

    with open(pathInput+'.json',) as file:
        data = json.load(file)

    G = nj.node_link_graph(data)

    return G

def saveGraphJson(nxGraph , pathOutput):

    # convert to list types
    for component in [nxGraph.edges(), nxGraph.nodes()]:
        for u in component:
            for k, v in component[u].items():
                if isinstance(v, np.ndarray):
                    component[u][k] = v.tolist()

    data = nj.node_link_data(nxGraph)
    with open(pathOutput+'.json', 'w+') as file:
        json.dump(data, file)
