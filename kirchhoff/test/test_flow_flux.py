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
import sys
import kirchhoff.circuit_flow as kfc
import kirchhoff.circuit_dual as kid

def test_circuit_plot2D():

    K = kfc.initialize_flow_circuit_from_crystal('hexagonal',3)
    K.set_source_landscape()
    K.set_plexus_landscape()

    fig=K.plot_circuit()

def test_circuit_plot3D():

    K=kfc.initialize_flow_circuit_from_crystal('simple',3)
    K.set_source_landscape()
    K.set_plexus_landscape()

    fig=K.plot_circuit()

def test_circuit_plotDual():

    D=kid.initialize_dual_from_minsurf(circuit_type='flow', dual_type='laves',num_periods=2)
    fig=D.plot_circuit()

    fig=D.layer[0].plot_circuit()
    fig=D.layer[1].plot_circuit()


def test_circuit_plotRandom():

    K=kfc.initialize_flow_circuit_from_random(random_type='voronoi_volume', periods=10, sidelength=10)
    K.set_source_landscape()
    K.set_plexus_landscape()

    fig=K.plot_circuit()
