# @Author:  Felix Kramer
# @Date:   2021-11-06T16:28:24+01:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-07T12:53:24+01:00
# @License: MIT
import networkx as nx
import numpy as np
import kirchhoff.circuit_init as ki
import kirchhoff.init_dual as kid
import kirchhoff.init_crystal as kic

import kirchhoff.circuit_flow as kfc
import kirchhoff.circuit_flux as kfx
import kirchhoff.circuit_dual as kcd

def test_circuit_constructors():

    n=3
    G=nx.grid_graph(( n,n,1))

    # circuits
    K = ki.Circuit(G)
    # print(K)
    K = ki.initialize_circuit_from_crystal('simple',3)
    # print(K)
    K = ki.initialize_circuit_from_random(random_type='voronoi_volume')
    # print(K)

    # flowcircuits
    K = kfc.FlowCircuit(G)
    # print(K)
    K = kfc.initialize_flow_circuit_from_crystal('simple',3)
    # print(K)
    K = kfc.initialize_flow_circuit_from_random(random_type='voronoi_volume')
    # print(K)

    # fluxcircuits
    K = kfx.FluxCircuit(G)
    # print(K)
    K = kfx.initialize_flux_circuit_from_crystal('simple',3)
    # print(K)
    K = kfx.initialize_flux_circuit_from_random(random_type='voronoi_volume')
    # print(K)

def test_grid():

    import kirchhoff.init_random as kir

    constructors = {
            'default': kir.NetworkxVoronoiPlanar,
            'voronoi_planar': kir.NetworkxVoronoiPlanar,
            'voronoi_volume': kir.NetworkxVoronoiVolume,
            }

    for k,constr in constructors.items():

        grid = kir.init_graph_from_random(k, periods=10, sidelength=10)
        print(nx.info(grid))

def test_circuit_dual():

    from kirchhoff.circuit_init import Circuit
    from kirchhoff.circuit_flow import FlowCircuit
    from kirchhoff.circuit_flux import FluxCircuit

    circuitConstructor = {
        'default' : Circuit,
        'circuit' : Circuit,
        'flow' : FlowCircuit,
        'flux' : FluxCircuit,
        }

    print('test from networkx')
    n=3
    G=nx.grid_graph(( n,n,1))

    for k, constr in circuitConstructor.items():

        circuitSet = [constr(G), constr(G)]
        K = kcd.DualCircuit(circuitSet)
        print(K.layer)

    print('test from catenation')
    for k, constr in circuitConstructor.items():

        K = kcd.initialize_dual_from_catenation(k, 'catenation', 3)
        print(K.layer)

    print('test from minsurf')
    for k, constr in circuitConstructor.items():

        K = kcd.initialize_dual_from_minsurf(k, 'simple', 3)
        print(K.layer)

def test_grid_dual():

    constructors = {
            'default': kid.NetworkxDualSimple,
            'simple': kid.NetworkxDualSimple,
            'diamond': kid.NetworkxDualDiamond,
            'laves': kid.NetworkxDualLaves,
        }

    for k,constr in constructors.items():

        grid = kid.init_dual_minsurf_graphs(k, 3)
        print(nx.info(grid.layer[0]))
        print(nx.info(grid.layer[1]))

    constructors = {
            'default': kid.NetworkxDualCatenation,
            'catenation': kid.NetworkxDualCatenation,
            'crossMesh': kid.NetworkxDualCrossMesh,
        }


    for k,constr in constructors.items():

        if k == 'crossMesh':
            n = [3, 3, 3, 3]
        else:
            n = 3

        grid = kid.init_dualCatenation(k, n)
        print(nx.info(grid.layer[0]))
        print(nx.info(grid.layer[1]))
