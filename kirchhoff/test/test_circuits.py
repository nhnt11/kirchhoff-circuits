# @Author:  Felix Kramer
# @Date:   2021-11-06T16:28:24+01:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-06T18:46:08+01:00
# @License: MIT
import networkx as nx
import numpy as np
import kirchhoff.circuit_init as ki


def test_circuit_nx():

    n = 3
    G = nx.grid_graph((n, n, 1))
    K = ki.initialize_circuit_from_networkx(G)

def test_circuit_crystal():

    pr = 5
    choose_constructor_option = {
        'default': networkx_simple,
        'simple': networkx_simple,
        'chain': networkx_chain,
        'bcc': networkx_bcc,
        'fcc': networkx_fcc,
        'diamond': networkx_diamond,
        'laves':networkx_laves,
        'trigonal_stack': networkx_trigonal_stack,
        'square': networkx_square,
        'hexagonal':networkx_hexagonal,
        'trigonal_planar':networkx_trigonal_planar,
        }

    for crystal in choose_constructor_option:
        K = ki.initialize_circuit_from_crystal(crystal_type=crystal, periods=pr)

def test_circuit_random():

    n = 3
    G = nx.grid_graph((n, n, 1))
    K = ki.initialize_circuit_from_random(random_type='default', periods=10, sidelength=1)

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
