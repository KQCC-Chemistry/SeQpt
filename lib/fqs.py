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
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.circuit.library.standard_gates.u3 import U3Gate

#from gate_parameter import GateParameter
from . import PlasticPQC

import math
import cmath
import pprint


class FreeQuaternionSelection():

    def free_quaternion_selection(self, idx=None, current_val=None, display=False):

        if current_val is None:
            current_val, _ = self._pqc.cost_evaluation(params=self._pqc._params, display=True)

        ## evaluate 4x4 matrix (real symmertic matrix S)
        matrix = self._fqs_matrix(self._pqc._params, idx)
        if display is True :
            print("   fqsopt real symmetric matrix S : \n{}".format(np.matrix(matrix)))

        eigVal, eigVec = np.linalg.eig(matrix)
        eigVec = np.transpose(eigVec)
        # print("(debug @fqs) eigVec : {}".format(eigVec))
        #print("   fqsopt eigVal : {}".format(eigVal))

        sid = np.argmin(eigVal)
        expected_val = np.amin(eigVal)

        if expected_val < current_val:
            self._pqc._params[idx]  = eigVec[sid]
            self._pqc._u3params[idx]= self._pqc.param_QtoU3(self._pqc._params[idx])
            check_val, _ = self._pqc.cost_evaluation(params=self._pqc._params)

        else :
            print('   Warning!! not updated')
            print('   candidate val = {}'.format(eigVal))
            print('   previous  val = {}'.format(current_val) )
        return expected_val

    def _fqs_matrix(self, params, gate_index):

        temporal_parm = copy.deepcopy(params)
        """theta  = params[index,0]: phi = params[index,1]: lamb = params[index,2]"""

        # 0,0 (MXrX) #
        temporal_parm[gate_index] =  [0,1,0,0]
        rx, _ = self._pqc.cost_evaluation(params=temporal_parm)

        # 1,1 (MYrY) #
        temporal_parm[gate_index] =  [0,0,1,0]
        ry, _ = self._pqc.cost_evaluation(params=temporal_parm)

        # 2,2 (MZrZ) #
        temporal_parm[gate_index] =  [0,0,0,1]
        rz, _ = self._pqc.cost_evaluation(params=temporal_parm)

        # 0,1 [M(X+Y)r(X+Y)] #
        temporal_parm[gate_index] =  [0, 1/np.sqrt(2), 1/np.sqrt(2), 0]
        rxy, _ = self._pqc.cost_evaluation(params=temporal_parm)

        # 1,2 [M(Y+Z)r(Y+Z)] #
        temporal_parm[gate_index] =  [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]
        ryz, _ = self._pqc.cost_evaluation(params=temporal_parm)

         # 3,1 [M(Z+X)r(Z+X)] #
        temporal_parm[gate_index] =  [0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]
        rzx, _ = self._pqc.cost_evaluation(params=temporal_parm)

        ## 0,0 id
        temporal_parm[gate_index] =  [1, 0, 0, 0]
        id, _ = self._pqc.cost_evaluation(params=temporal_parm)

        ## 0,1 bxpls
        temporal_parm[gate_index] =  [1/np.sqrt(2), 1/np.sqrt(2), 0, 0]
        bxpls, _ = self._pqc.cost_evaluation(params=temporal_parm)

        ## 0,2 bypls
        temporal_parm[gate_index] =  [1/np.sqrt(2),0, 1/np.sqrt(2), 0]
        bypls, _ = self._pqc.cost_evaluation(params=temporal_parm)

         ## 0,3 bzpls
        temporal_parm[gate_index] =  [1/np.sqrt(2),0,0, 1/np.sqrt(2)]
        bzpls, _ = self._pqc.cost_evaluation(params=temporal_parm)

        matrix=[[id             , bxpls-rx/2-id/2, bypls-ry/2-id/2, bzpls-rz/2-id/2],
                [bxpls-rx/2-id/2,  rx            , (2*rxy-rx-ry)/2, (2*rzx-rx-rz)/2],
                [bypls-ry/2-id/2, (2*rxy-rx-ry)/2,  ry            , (2*ryz-ry-rz)/2],
                [bzpls-rz/2-id/2, (2*rzx-rx-rz)/2, (2*ryz-ry-rz)/2,  rz            ]]

        return matrix



    def compare2Identity(self, gate_idx, normtype="Frobenius"):
    
        acc  = 0
        acc2 = 0
        acc_matrix = np.zeros((4,4), dtype=float)
        I_matrix = np.identity(4)
        for n in range(self._max_iteration):
#            S    = self._random_S(gate_idx, display)
            params   = self._gen_params()
            self._ref_state = random_statevector(dims=2**self._num_qubits, seed=n)

            matrix   = self._matrix_evaluation(self._params, gate_idx)

            # Frobenius norm
            if normtype.lower() == "frobenius" :
                norm = np.linalg.norm(matrix)/2
                matrix = matrix/norm
#                print(matrix)
                acc_matrix = acc_matrix+matrix
                # L1 = np.linalg.norm(matrix - I_matrix)/2
                L1 = np.linalg.norm(matrix - I_matrix)
                acc  = acc  + L1
                acc2 = acc2 + L1*L1

#            print(S)
            # norm = np.linalg.norm(S,'nuc')
            #norm = np.sqrt(np.sqrt(abs(np.linalg.det(S))))
            # print(norm)
            #normalized_S = S/norm
            #norm_list.append(np.linalg.norm(normalized_S-np.identity(4)))

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
#        return norm_list


