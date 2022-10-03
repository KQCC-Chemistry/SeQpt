
from typing import Optional, List, Callable, Union, Dict, Any
import logging
import warnings
import time
import numpy as np
import copy

from qiskit import  QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
import math 

logger = logging.getLogger(__name__)


def QmoleculeProcessing(qmolecule, pACTIVESPACE ) : 
    num_particles       = qmolecule.num_alpha+qmolecule.num_beta
    num_spin_orbitals   = qmolecule.num_orbitals*2
    print("# of electrons: {}".format(num_particles))
    print("# of spatial orbitals: {}".format(qmolecule.num_orbitals))
    print("# of spin orbitals: {}".format(num_spin_orbitals))
    print("# of num_alpha: {}".format(qmolecule.num_alpha))
    
    active_space = [x + qmolecule.num_alpha-1 for x in pACTIVESPACE]
    print("active space {}".format(active_space))
    orbital_reduction = [x for x in range(qmolecule.num_orbitals) if x not in active_space ]
    print("removed orbital {}".format(orbital_reduction))
    
    num_AS_spin_orbitals = len(active_space)*2
    num_AS_particles     = len([x for x in active_space if x<qmolecule.num_alpha ])*2
    
    print("number of spin orbitals in active space: {}".format(num_AS_spin_orbitals))
    print("number of electrons in active space: {}".format(num_AS_particles))
   
    return num_AS_spin_orbitals, num_AS_particles, orbital_reduction
