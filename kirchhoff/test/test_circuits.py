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

def test_circuit_nx():

    n = 3
    G = nx.grid_graph((n, n, 1))
    K = ki.initialize_circuit_from_networkx(G)

def test_circuit_crystal():

    pr = 3
    choose_constructor_option = [
        'default',
        'simple',
        'chain',
        'bcc',
        'fcc',
        'diamond',
        'laves',
        'square',
        'hexagonal',
        'trigonal_planar'
        ]

    for crystal in choose_constructor_option:
            K = kic.init_graph_from_crystal(crystal_type=crystal, periods=pr)

    choose_constructor_option = [
        'trigonal_stack'
        ]

    for crystal in choose_constructor_option:
            K = kic.init_graph_from_asymCrystal(crystal_type=crystal, periodsZ=pr, periodsXY=pr)

def test_circuit_random():

    choose_constructor_option={
        'default',
        'voronoi_planar',
        'voronoi_volume'
        }

    for rand in choose_constructor_option:
        K = ki.initialize_circuit_from_random(random_type=rand, periods=10, sidelength=1)

def test_circuit_dual():

    pr = 3
    choose_constructor_option = [
        'simple',
        'diamond',
        'laves'
        ]
    for dual in choose_constructor_option:
        D = kid.init_dual_minsurf_graphs(dual_type=dual, num_periods=pr)

    choose_constructor_option = [
        'catenation'
        ]
    for dual in choose_constructor_option:
        D = kid.init_dual_catenation(dual_type=dual, num_periods=pr)

# def test_circuit_flow():
#
#     import kirchhoff.circuit_flow as kfc
#     kfc.initialize_circuit_from_networkx(G)
#     kfc.initialize_flow_circuit_from_crystal('simple', 3)
#     kfc.initialize_flow_circuit_from_random(random_type='voronoi_volume')
#
# def test_circuit_flux():
#
#     import kirchhoff.circuit_flux as kfx
#     kfx.initialize_circuit_from_networkx(G)
#     kfx.initialize_flux_circuit_from_crystal('simple', 3)
#     kfx.initialize_flux_circuit_from_random(random_type='voronoi_volume')
#
# def test_circuit_dual():
#
#     import kirchhoff.circuit_dual as kid
#     kid.initialize_dual_flux_circuit_from_minsurf('simple', 3)
