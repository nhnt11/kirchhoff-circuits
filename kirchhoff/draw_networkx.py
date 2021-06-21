# @Author:  Felix Kramer <kramer>
# @Date:   2021-05-08T20:34:30+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-05-23T23:17:48+02:00
# @License: MIT

# standard types
import networkx as nx
import numpy as np
import plotly.graph_objects as go

#generate interactive plots with plotly and return the respective figures
def plot_networkx(input_graph,**kwargs):

    options={
        'network_id':0,
        'color_nodes':['#a845b5'],
        'color_edges':['#c762d4']
    }
    for k,v in kwargs.items():
        if k in options:
            options[k]=v

    fig = go.Figure()
    plot_nodes_edges(fig,options,input_graph)
    fig.update_layout(showlegend=False)

    return fig

def plot_networkx_dual(dual_graph,**kwargs):

    options={
        'network_id':0,
        'color_nodes':['#6aa84f','#a845b5'],
        'color_edges':['#2BDF94','#c762d4']
    }
    for k,v in kwargs.items():
        if k in options:
            options[k]=v

    fig = go.Figure()
    for i,K in enumerate(dual_graph.layer):
        options['network_id']=i
        plot_nodes_edges(fig,options,K.G)
    fig.update_layout(showlegend=False)

    return fig

#auxillary functions generating traces for nodes and edges
def get_edge_coords(input_graph,options):

    pos=nx.get_node_attributes(input_graph,'pos')

    if len(list(pos.values())[0]) != options['dim']:
        options['dim']=len(list(pos.values())[0])
    edge_xyz = [[] for i in range(options['dim'])]

    for edge in input_graph.edges():

        xyz_0= pos[edge[0]]
        xyz_1 = pos[edge[1]]

        for i in range(options['dim']):

            edge_xyz[i].append(xyz_0[i])
            edge_xyz[i].append(xyz_1[i])
            edge_xyz[i].append(None)

    return edge_xyz

def get_edge_scatter(edge_xyz,options):

    if options['dim']==3:
        edge_trace = go.Scatter3d(
            x=edge_xyz[0], y=edge_xyz[1],z=edge_xyz[2],
            line=dict(width=5, color=options['color']),
            hoverinfo='none',
            mode='lines')
    else:
        edge_trace = go.Scatter(
            x=edge_xyz[0], y=edge_xyz[1],
            line=dict(width=5, color=options['color']),
            hoverinfo='none',
            mode='lines')

    return edge_trace

def get_edge_trace(input_graph, **kwargs):

    options={
        'color':'#888',
        'dim':3
    }
    for k,v in kwargs.items():
        if k in options:
            options[k]=v

    edge_xyz=get_edge_coords(input_graph,options)
    edge_trace=get_edge_scatter(edge_xyz,options)

    return edge_trace

def get_node_coords(input_graph,options):

    pos=nx.get_node_attributes(input_graph,'pos')
    if len(list(pos.values())[0])!=options['dim']:
        options['dim']=len(list(pos.values())[0])

    node_xyz = [[] for i in range(options['dim'])]
    for node in input_graph.nodes():

        xyz_0= pos[node]

        for i in range(options['dim']):

            node_xyz[i].append(xyz_0[i])

    return node_xyz

def get_node_scatter(node_xyz,options):

    if options['dim']==3:
        node_trace = go.Scatter3d(
        x=node_xyz[0], y=node_xyz[1],z=node_xyz[2],
        mode='markers',
        hoverinfo='none',
        marker=dict(
            size=2,
            line_width=2,
            color=options['color'])
            )
    else:
        node_trace = go.Scatter(
        x=node_xyz[0], y=node_xyz[1],
        mode='markers',
        hoverinfo='none',
        marker=dict(
            size=2,
            line_width=2,
            color=options['color'])
            )

    return node_trace

def get_node_trace(input_graph,**kwargs):

    options={
        'color':'#888',
        'dim':3
    }
    for k,v in kwargs.items():
        if k in options:
            options[k]=v

    node_xyz=get_node_coords(input_graph,options)

    node_trace = get_node_scatter(node_xyz,options)

    return node_trace

# integrate traces into the figure
def plot_nodes_edges(fig,options,input_graph):

    idx=options['network_id']
    edge_trace=(get_edge_trace(input_graph,color=options['color_edges'][idx]))
    node_trace=(get_node_trace(input_graph,color=options['color_nodes'][idx]))
    fig.add_trace( edge_trace)
    fig.add_trace( node_trace)
