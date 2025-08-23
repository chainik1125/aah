"""
Helper functions to extract the spectrum in QuSpin.
"""

import numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian, quantum_operator
from quspin.basis import spinful_fermion_basis_1d
from itertools import combinations


