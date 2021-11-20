# @Author:  Felix Kramer <kramer>
# @Date:   2021-05-08T20:34:30+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-07T16:21:36+01:00
# @License: MIT

# standard types
import networkx as nx
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go


#generate interactive plots with plotly and return the respective figures
def plot_networkx(input_graph, **kwargs):

    """
    Return an interactive network plot, which shows the internal edge and node
     data via hover and text.

    Args:
        input_graph (nx.Graph): A networkx graph.
        kwargs (dictionary): A dictionary for plotly keywords customizing the plots' layout.

    Returns:
        plotly.graph_objects.Figure: A plotly figure object

    """

    options = {
        'network_id':0,
        'color_nodes':['#a845b5'],
        'color_edges':['#c762d4'],
        'colormap':['plasma'],
        'markersize': [2],
        'linewidth': [5],
        'axis': True
    }

    node_data = pd.DataFrame()
    edge_data = pd.DataFrame()

    for k, v in kwargs.items():
        if k in options:
            options[k] = v
    if 'node_data' in kwargs:
        for sk, sv in kwargs['node_data'].items():
            node_data[sk] = sv.to_numpy()
    if 'edge_data' in kwargs:
        for sk, sv in kwargs['edge_data'].items():
            edge_data[sk] = sv.to_numpy()

    fig = go.Figure()
    add_traces_nodes(fig, options, input_graph, node_data)
    add_traces_edges(fig, options, input_graph, edge_data)

    new_layout = {
        'showlegend': False,
        'autosize': True,
        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
        'scene': dict(aspectmode="data",)
    }
    if not options['axis']:
        new_layout.update(
            {'scene': dict(
                    xaxis_title='',
                    yaxis_title='',
                    zaxis_title='',
                    xaxis = dict(
                                showbackground=False,
                                showticklabels=False
                        ),
                    yaxis = dict(
                                showbackground=False,
                                showticklabels=False
                        ),
                    zaxis = dict(
                                showbackground=False,
                                showticklabels=False
                        ),
                    aspectmode="data",
                )
            }
        )

    fig.update_layout(**new_layout)

    return fig

def plot_networkx_dual(dual_circuit, *args, **kwargs):
    """
    Return an interactive network plot, which shows the internal edge and node
     data via hover and text for a multilayer system.

    Args:
        dual_circuit (dual_circuit): A dual_circuit object.
        args (list): A list for keywoprd data for display.
        kwargs (dictionary): A dictionary for plotly keywords customizing the plots' layout.

    Returns:
        plotly.graph_objects.Figure: A plotly figure object

    """

    options = {
        'network_id':0,
        'color_nodes':['#6aa84f', '#a845b5'],
        'color_edges':['#2BDF94', '#c762d4'],
        'colormap':['plasma', 'plasma'],
        'markersize': [2, 2],
        'linewidth': [10, 10],
        'axis': True
    }

    for k, v in kwargs.items():
        if k in options:
            options[k] = v

    fig = go.Figure()

    for i, K in enumerate(dual_circuit.layer):

        options['network_id'] = i
        node_data = K.get_nodes_data(*args)
        edges_data = K.get_edges_data(*args)

        add_traces_nodes(fig, options, K.G, node_data)
        add_traces_edges(fig, options, K.G, edges_data)

    new_layout = {
        'showlegend': False,
        'autosize': True,
        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
        'scene_camera': dict(eye=dict(x=2, y=2, z=0.9))
    }
    if not options['axis']:
        new_layout.update(
            {'scene': dict(
                    xaxis_title='',
                    yaxis_title='',
                    zaxis_title='',
                    xaxis = dict(
                                showbackground=False,
                                showticklabels=False,
                                autorange=True,
                        ),
                    yaxis = dict(
                                showbackground=False,
                                showticklabels=False,
                                autorange=True,
                        ),
                    zaxis = dict(
                                showbackground=False,
                                showticklabels=False,
                                autorange=True,
                        ),
                    aspectmode="data",
                )
            }
        )

    fig.update_layout(**new_layout)


    return fig

# integrate traces into the figure
def add_traces_edges(fig, options, input_graph, extra_data):

    """
    Add line traces for interactive edge data display.

    Args:
        fig (plotly.graph_objects.Figure: A plotly figure object
        options (dictionary): A dictionary for plotly keywords customizing the plots' layout.
        input_graph (nx.Graph): A networkx graph.
        extra_data (pandas.DataFrame): A dataframe holding the data.

    """

    idx = options['network_id']

    optM = {
        'color': options['color_edges'][idx],
        'colormap': options['colormap'][idx]
    }
    edge_mid_trace = get_edge_mid_trace(input_graph, extra_data, **optM)

    optI = {
        'color': options['color_edges'][idx],
        'linewidth': options['linewidth'][idx],
        'colormap': options['colormap'][idx]
    }

    edge_invd_traces = get_edge_invd_traces(input_graph, extra_data, **optI)

    for eit in edge_invd_traces:
        fig.add_trace(eit)

    fig.add_trace(edge_mid_trace)

def add_traces_nodes(fig,  options,  input_graph, extra_data):

    """
    Add point traces for interactive node data display.

    Args:
        fig (plotly.graph_objects.Figure: A plotly figure object
        options (dictionary): A dictionary for plotly keywords customizing the plots' layout.
        input_graph (nx.Graph): A networkx graph.
        extra_data (pandas.DataFrame): A dataframe holding the data.

    """

    idx = options['network_id']
    optN = {
        'color': options['color_nodes'][idx],
        'markersize': options['markersize'][idx],
        'colormap': options['colormap'][idx]
    }

    node_trace = get_node_trace( input_graph,  extra_data, **optN )
    fig.add_trace( node_trace)

#auxillary functions generating traces for nodes and edges
def get_edge_mid_trace(input_graph, extra_data,  **kwargs):

    """
    Return transparent line traces for interactive edge data display.

    Args:
        input_graph (nx.Graph): A networkx graph.
        extra_data (pandas.DataFrame): A dataframe holding the data.
        kwargs (dictionary): A dictionary for plotly keywords customizing the plots' layout.

    Returns:
        plotly.graph_objects.Scatter: A plotly scatter object

    """

    options = {
        'color':'#888',
        # 'dim':3
    }
    dim = 3
    for k, v in kwargs.items():
        if k in options:
            options[k] = v

    pos = nx.get_node_attributes(input_graph, 'pos')
    if len(list(pos.values())[0]) !=  dim:
        dim = len(list(pos.values())[0])

    E = input_graph.edges()
    # if 'edge_list' in options:
    #     E = options['edge_list']

    middle_node_trace = get_hover_scatter_from_template(dim, options)

    XYZ = [[] for i in range(dim)]
    for j, edge in enumerate(E):

        XYZ_0 = pos[edge[0]]
        XYZ_1 = pos[edge[1]]

        for i, xi in enumerate(XYZ):
            xi.append((XYZ_0[i]+XYZ_1[i])/2.)

    set_hover_info(middle_node_trace, XYZ, extra_data)

    return middle_node_trace

def set_hover_info(trace, XYZ, extra_data):

    """
    Set hover info for figure traces.

    Args:
        trace (plotly.graph_objects.trace): A networkx graph.
        XYZ (ndarray): Nodal position data.
        extra_data (pandas.DataFrame): A dataframe holding the data.

    """

    tags = ['x', 'y', 'z']
    if len(XYZ)<3:
        tags = ['x', 'y']
    for i, t in enumerate(tags):
        trace[t] = XYZ[i]

    if len(extra_data.keys())!= 0:
        data = [list(extra_data[c]) for c in extra_data.columns]
        iter = list(zip(*data))
        text = [create_tag(vals, extra_data.columns ) for vals in iter]
        trace['text'] = text
    else:
        trace['hoverinfo'] = 'none'

    trace['hoverlabel'] = dict(bgcolor="white")

def get_hover_scatter_from_template(dim, options):

    """
    Get scatter hover info for figure traces.

    Args:
        dim (int): A dimensional identifier.
        options (dictionary): A dictionary for plotly keywords customizing the plots' layout.

    Returns:
        plotly.graph_objects.Scatter: A plotly scatter object

    """

    if dim == 3:
        middle_node_trace = go.Scatter3d(
            x = [],
            y = [],
            z = [],
            text = [],
            mode = 'markers',
            hoverinfo = 'text',
            opacity = 0,
            marker = dict(**options)
            # marker = dict(color = options['color'])
        )
    else:
        middle_node_trace  =  go.Scatter(
            x = [],
            y = [],
            text = [],
            mode = 'markers',
            hoverinfo = 'text',
            marker = go.scatter.Marker(
                opacity = 0,
                **options
            )

        )

    return middle_node_trace

def get_edge_invd_traces(input_graph, extra_data,  **kwargs):

    """
    Return individual line traces for interactive edge data display.

    Args:
        input_graph (nx.Graph): A networkx graph.
        extra_data (pandas.DataFrame): A dataframe holding the data.
        kwargs (dictionary): A dictionary for plotly keywords customizing the plots' layout.

    Returns:
        plotly.graph_objects.Scatter: A plotly scatter object

    """

    options = {
        'color':'#888',
        'width': 2,
        # 'dim':3
    }
    # options['width'] = kwargs['linewidth']
    dim = 3
    for k, v in kwargs.items():
        if k in options:
            options[k] = v

    # handle exceptions and new containers
    colorful = False
    if type(options['color']) != str:
        colorful = True
        cmax = np.max(options['color'])
        # cmin = np.min(options['color'])
        if cmax == 0:
            cmax=1.

        pc = plotly.colors.sample_colorscale(kwargs['colormap'], options['color']/cmax)
        options['color'] = pc

    weighted = False
    if 'linewidth' in kwargs.keys():
        options['width']=kwargs['linewidth']
    if type(options['width']) != int:
        weighted = True

    pos = nx.get_node_attributes(input_graph, 'pos')
    if len(list(pos.values())[0]) != dim:
        dim = len(list(pos.values())[0])

    E = input_graph.edges()
    # if 'edge_list' in options:
    #     E = options['edge_list']

    # add new traces
    trace_list = []
    aux_option = dict(options)

    for i, edge in enumerate(E):

        if colorful:
            aux_option['color'] = options['color'][i]

        if weighted:
            aux_option['width'] = options['width'][i]

        trace = get_line_from_template(dim, aux_option)
        XYZ_0 = input_graph.nodes[edge[0]]['pos']
        XYZ_1 = input_graph.nodes[edge[1]]['pos']

        set_edge_info(trace, XYZ_0, XYZ_1)
        trace_list.append(trace)

    return trace_list

def set_edge_info(trace, XYZ_0, XYZ_1):

    """
    Set hover info for figure traces.

    Args:
        trace (plotly.graph_objects.trace): A networkx graph.
        XYZ_0 (ndarray): Nodal position data.
        XYZ_0 (ndarray): Nodal position data.
    """

    tags = ['x', 'y', 'z']
    if len(XYZ_0)<3:
        tags = ['x', 'y']

    for i, t in enumerate(tags):
        trace[t] = [XYZ_0[i],  XYZ_1[i],  None]

def get_line_from_template(dim, options):

    """
    Get a line trace element.

    Args:
        dim (int): A dimensional identifier.
        options (dictionary): A dictionary for plotly keywords customizing the plots' layout.

    Returns:
        plotly.graph_objects.Scatter: A plotly scatter object

    """

    if dim == 3:

        trace = go.Scatter3d(
            x = [],
            y = [],
            z = [],
            mode = 'lines',
            line = dict(**options),
            hoverinfo = 'none'
        )

    else:

        trace = go.Scatter(
            x = [],
            y = [],
            mode = 'lines',
            line = dict(**options),
            hoverinfo = 'none'
        )

    return  trace

def get_node_trace(input_graph, extra_data, **kwargs):


    """
    Return nodal traces for interactive node data display.

    Args:
        input_graph (nx.Graph): A networkx graph.
        extra_data (pandas.DataFrame): A dataframe holding the data.
        kwargs (dictionary): A dictionary for plotly keywords customizing the plots' layout.

    Returns:
        plotly.graph_objects.Scatter: A plotly scatter object

    """

    options = {
        'color':'#888',
        'dim':3,
        'markersize':2
    }

    for k, v in kwargs.items():
        if k in options:

            options[k] = v

    node_xyz = get_node_coords(input_graph, options)

    node_trace = get_node_scatter(node_xyz, extra_data, options)

    return node_trace

def get_node_coords(input_graph, options):


    """
    Return nodal coordinates in specified tuple format.

    Args:
        input_graph (nx.Graph): A networkx graph.
        options (dictionary): A dictionary for circuit keywords for readout.

    Returns:
        list: A list of nodal postions [X, Y, Z]

    """

    pos = nx.get_node_attributes(input_graph, 'pos')
    if len(list(pos.values())[0])!= options['dim']:
        options['dim'] = len(list(pos.values())[0])

    node_xyz = [[] for i in range(options['dim'])]

    N = input_graph.nodes()
    if 'node_list' in options:
        N = options['edge_list']

    for node in N:

        xyz_0 =  pos[node]

        for i in range(options['dim']):

            node_xyz[i].append(xyz_0[i])

    return node_xyz

def get_node_scatter(node_xyz, extra_data, options):

    """
    Return nodal traces for interactive node data display.

    Args:
        node_xyz (list): A list of nodal postions [X, Y, Z].
        extra_data (pandas.DataFrame): A dataframe holding the data.
        options (dictionary): A dictionary for plotly keywords customizing the plots' layout.

    Returns:
        plotly.graph_objects.Scatter: A plotly scatter object

    """

    mode = 'none'
    hover = ''

    if len(extra_data.keys()) != 0:
        mode = 'text'
        data = [list(extra_data[c]) for c in extra_data.columns]
        iter = list(zip(*data))
        hover = [create_tag(vals, extra_data.columns ) for vals in iter]

    if options['dim'] == 3:
        node_trace = go.Scatter3d(
        x=node_xyz[0], y=node_xyz[1], z=node_xyz[2],
        mode='markers',
        hoverinfo=mode,
        hovertext=hover,
        marker=dict(
            size=options['markersize'],
            line_width=2,
            color=options['color'])
        )
    else:
        node_trace = go.Scatter(
        x=node_xyz[0], y=node_xyz[1],
        mode='markers',
        hoverinfo=mode,
        hovertext=hover,
        marker=dict(
            size=options['markersize'],
            line_width=2,
            color=options['color'])
            )

    return node_trace

def create_tag(vals, columns):

    """
    Create a hover tag.

    Args:
        vals (list): A list of values.
        columns (list):A list of keywords.

    Returns:
        string: A string in makrdown format.

    """

    tag = f''
    for i, c in enumerate(columns):
        tag+= str(c)+': '+str(vals[i])+'<br>'

    return tag
