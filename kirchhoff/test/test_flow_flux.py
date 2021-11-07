# @Author:  Felix Kramer
# @Date:   2021-11-07T12:20:58+01:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-07T12:54:16+01:00
# @License: MIT
# @Author:  Felix Kramer
# @Date:   2021-11-06T16:28:24+01:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-07T12:54:16+01:00
# @License: MIT
import networkx as nx
import numpy as np
import kirchhoff.circuit_flow as kfc
import kirchhoff.circuit_flux as kfx
import kirchhoff.circuit_dual as kid



def test_circuit_flow_flux():

    # fixed networkx graph
    n = 3
    G = nx.grid_graph((n, n, 1))
    kfc.initialize_circuit_from_networkx(G)
    kfx.initialize_circuit_from_networkx(G)

    # custom crystal contructor
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
        kfc.initialize_flow_circuit_from_crystal(crystal, 3)
        kfx.initialize_flux_circuit_from_crystal(crystal, 3)

    # custom random contructor
    choose_constructor_option={
        'default',
        'voronoi_planar',
        'voronoi_volume'
        }

    for rand in choose_constructor_option:
        opts= {
            'random_type': rand,
            'periods': 10,
            'sidelength': 1
        }
        kfc.initialize_flow_circuit_from_random(**opts)
        kfx.initialize_flux_circuit_from_random(**opts)

def test_circuit_dual():


    # fixed networkx graph
    n = 3
    G = nx.grid_graph((n, n, 1))

    D = kid.dual_circuit()
    D.circuit_init_from_networkx([G, G])

    kid.initialize_dual_flow_circuit_from_networkx(G, G, [])
    kid.initialize_dual_flux_circuit_from_networkx(G, G, [])

    # custom crystal contructor
    choose_constructor_option = [
        'simple',
        'diamond',
        'laves'
        ]
    for dual in choose_constructor_option:

        kid.initialize_dual_flow_circuit_from_minsurf(dual, 3)
        kid.initialize_dual_flux_circuit_from_minsurf(dual, 3)

    choose_constructor_option = [
        'catenation'
        ]

    for dual in choose_constructor_option:

        kid.initialize_dual_flow_circuit_from_catenation(dual, 3)
        kid.initialize_dual_flux_circuit_from_catenation(dual, 3)
