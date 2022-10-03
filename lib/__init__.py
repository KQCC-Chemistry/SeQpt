# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=====================================
Algorithms (:mod:`qiskit.algorithms`)
=====================================

Phase Estimators
----------------

Algorithms that estimate the phases of eigenstates of a unitary.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   HamiltonianPhaseEstimation
   HamiltonianPhaseEstimationResult
   PhaseEstimationScale
   PhaseEstimation
   PhaseEstimationResult
   IterativePhaseEstimation

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   AlgorithmError
"""

from .realdevice import RealDeviceSelector
from .circuitdesign import CircuitDesigner, EntanglerDesigner, ZYcircuit
from .gate_parameter import GateParameter
from .plastic_pqc import PlasticPQC
from .roto import RotationOf
from .fraxis import FreeAxisSelection
from .fqs import FreeQuaternionSelection
from .sequential_optimizer import SequentialOptimizer
from .qubit_op_library import QubitOpLibrary



__all__ = [
    "CircuitDesigner",
    "ZYcircuit",
    "LayerCreator",
    "EntanglerDesigner",
    "RealDeviceSelector",
    "GateParameter",
    "ParameterizedU3",
    "RotationOf",
    "FreeAxisSelection",
    "FreeQuaternionSelection",
    "SequentialOptimizer",
    "QubitOpLibrary",
]
