# @Author: Felix Kramer <kramer>
# @Date:   03-07-2022
# @Email:  felixuwekramer@proton.me
# @Last modified by:   kramer
# @Last modified time: 08-07-2022


import networkx.readwrite.json_graph as nj
import json
import numpy as np


def loadGraphJson(pathInput):

    with open(pathInput+'.json',) as file:
        data = json.load(file)

    G = nj.node_link_graph(data)

    return G


def saveGraphJson(nxGraph, pathOutput):

    # convert to list types
    for component in [nxGraph.edges(), nxGraph.nodes()]:
        for u in component:
            for k, v in component[u].items():
                if isinstance(v, np.ndarray):
                    component[u][k] = v.tolist()

    data = nj.node_link_data(nxGraph)
    with open(pathOutput+'.json', 'w+') as file:
        json.dump(data, file)
