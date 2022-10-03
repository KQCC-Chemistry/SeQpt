from typing import Optional, List, Callable, Union, Dict, Any
import logging
import warnings
import time
import numpy as np
import copy
import itertools

from qiskit import  QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
#from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.expectations import expectation_factory
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.quantum_info import Statevector, random_statevector
#from qiskit.quantum_info.states import Statevector
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.circuit.library.standard_gates.u3 import U3Gate

#from gate_parameter import GateParameter
from .plastic_pqc import PlasticPQC

import math
import cmath
import pprint


class FreeAxisSelection():

    def free_axis_selection(self, idx=None, current_val=None):

        if current_val is None:
            current_val, _ = self._pqc.cost_evaluation(self._pqc._params)
        print("current_val   :", current_val)
        measured_value, _ = self._pqc.cost_evaluation(self._pqc._params)
        print("measured_value:", measured_value)

        matrix = self._fraxis_matrix(idx)
        eigVal, eigVec = np.linalg.eig(matrix)
        eigVec = np.transpose(eigVec)

        sid = np.argmin(eigVal)
        expected_val = np.amin(eigVal)*0.5

        u3 = self._pqc.param_CtoU(eigVec[sid], np.pi)
        quaternion = [0, eigVec[sid][0], eigVec[sid][1], eigVec[sid][2]] 

        if expected_val < current_val:
            self._pqc._params[idx] = quaternion
        else :
            print('Warning!! not updated')
            print('eigeva_val', eigVal*0.5)

        print('expecet_val=', expected_val)
        return expected_val



    def _fraxis_matrix(self, gate_index) -> 'VQResult':

        temporal_parm = copy.deepcopy(self._pqc._params)
        # 0,0 Tr[MXrX] #
#        temporal_parm[gate_index] =  [np.pi, 0.0, np.pi]
        temporal_parm[gate_index] =  [0, 1, 0, 0]
        rx, _ = self._pqc.cost_evaluation(params=temporal_parm)

        # 1,1 Tr[MYrY] #
#        temporal_parm[gate_index] =  [np.pi, np.pi*0.5, np.pi*0.5]
        temporal_parm[gate_index] =  [0, 0, 1, 0]
        ry, _ = self._pqc.cost_evaluation(params=temporal_parm)

        # 2,2 Tr[MZrZ] #
#        temporal_parm[gate_index] =  [0.0, 0.0, np.pi]
        temporal_parm[gate_index] =  [0, 0, 0, 1]
        rz, _ = self._pqc.cost_evaluation(params=temporal_parm)

        # 0,1 Tr[M(X+Y)r(X+Y)] #
#        temporal_parm[gate_index] =  [np.pi, np.pi*0.25, np.pi*0.75]
        temporal_parm[gate_index] =  [0, 1/np.sqrt(2), 1/np.sqrt(2), 0]
        rxy, _ = self._pqc.cost_evaluation(params=temporal_parm)

        # 1,2 Tr[M(Y+Z)r(Y+Z)] #
#        temporal_parm[gate_index] =  [np.pi*0.5, np.pi*0.5, np.pi*0.5]
        temporal_parm[gate_index] =  [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]
        ryz, _ = self._pqc.cost_evaluation(params=temporal_parm)

         # 3,1 Tr[M(Z+X)r(Z+X)] #
#        temporal_parm[gate_index] =  [np.pi*0.5, 0.0, np.pi]
        temporal_parm[gate_index] =  [0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]
        rzx, _ = self._pqc.cost_evaluation(params=temporal_parm)

#        print(rx, ry, rz, rxy, ryz, rzx)
        matrix=[[2*rx,        2*rxy-rx-ry, 2*rzx-rx-rz],
                [2*rxy-rx-ry,        2*ry, 2*ryz-ry-rz],
                [2*rzx-rx-rz, 2*ryz-ry-rz,        2*rz]]

#        pprint.pprint("Matrix {}".format(matrix))
        return matrix


    def compare2Identity(self, gate_idx, normtype="Frobenius"):

        acc  = 0
        acc2 = 0
        acc_matrix = np.zeros((3,3), dtype=float)
        I_matrix = np.identity(3)
        print('max_iteration     : {}'.format(self._max_iteration))
        for n in range(self._max_iteration) :
            self._cparam = self.gen_fraxis_cparam()
            self._params = self.paramset_CtoU(self._cparam, self._angle, self._num_gates)
            self._ref_state = random_statevector(dims=2**self._num_qubits, seed=n)

            matrix   = self.matrix_evaluation(self._params, gate_idx)

            # Frobenius norm
            if normtype.lower() == "frobenius":
                norm = np.linalg.norm(matrix)/math.sqrt(3)
                matrix = matrix/norm
                acc_matrix = acc_matrix+matrix
                # L1 = np.linalg.norm(matrix - I_matrix)/2
                L1 = np.linalg.norm(matrix - I_matrix)
                acc  = acc  + L1
                acc2 = acc2 + L1*L1

        mean = acc/self._max_iteration
        var  = acc2/self._max_iteration - mean*mean
        acc_matrix = acc_matrix/self._max_iteration
        print("averaged Fraxis matrix")
        print(acc_matrix)
        content =':::'
        content += format(self._max_iteration, '5d')+ '  '
        content += format(self._depth, '5d')+ '  '
        content += format(mean, '8f')+ '  '
        content += format(var, '.4f')
        print("   trial  depth      mean     var", flush=True)
        print(content, flush=True)
        with open(self._output_name, 'a') as fp:
           print("   trial  depth      mean     var", file=fp, flush=True)
           print(content, file=fp, flush=True)

        return

 #This method is called when cparam is None. In Fraxis, angle=Pi, thus only cparam determination is required.
    def gen_fraxis_cparam(self, random_type: str = "state-random"):

        if  random_type.lower() == "parameter-random":
            cparam = self.gen_cparam_ParamRandom(self._num_gates)
        elif random_type.lower() == "state-random":
            cparam = self.gen_cparam_StateRandom(self._num_gates)
        elif random_type.lower() == "perturb":
            cparam = self.gen_cparam_perturb(num_self._num_gates)
        else:
            raise TypeError("GateParameter.gen_fraxis_param: initial_type is either param-random, state-random or perturb")
        return cparam


