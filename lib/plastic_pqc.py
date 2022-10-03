# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Variational Quantum Eigensolver algorithm.

See https://arxiv.org/abs/1304.3061
"""

from typing import Optional, List, Callable, Union, Dict, Any
import logging
import warnings
import time
import numpy as np
import copy
import itertools

from qiskit import  QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, U3Gate
#from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliExpectation
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.expectations import ExpectationFactory
from qiskit.opflow.expectations.expectation_base import ExpectationBase
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.opflow.state_fns.vector_state_fn import VectorStateFn
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.converters import CircuitSampler
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info import Statevector, random_statevector
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.algorithms import NumPyEigensolver, EigensolverResult

from .gate_parameter   import GateParameter
from .statevector_ex import StatevectorEx

import math 
import cmath 
import pprint

logger = logging.getLogger(__name__)

class PlasticPQC(GateParameter):
    """
    Arg:
       gate_spot(str) 
          * outer: standard ansatz structure,  alternating structure of single-qubit gate and controlled-gate layers.
          * inner: benchmark ansatz,  alternating controlled- and single-qubit gates.
          * zero: specific ansatz,  used only to construcut an ansatz: |0> + entangling layer  + single-qubit layer.


    """

    def __init__(self,
                 num_qubits      : int = None, 
                 params          : np.ndarray = None,
                 gate_list       : np.ndarray = None,
                 qubit_op        : OperatorBase = None,
                 quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
                 pre_circuit     : QuantumCircuit = None,
                 post_circuit    : QuantumCircuit = None,
                 control_type    : Optional[str] = "x" ,
                 cost_type       : Optional[str] = "Expectation" ,
                 initial_type    : Optional[str] = 'random',
                 ref_state       : Optional[QuantumState]=None
                 ) -> None:
        super().__init__(num_qubits)

        if quantum_instance is None :
             raise TypeError("quantum_instance is not given!")

        if gate_list is None :
             raise TypeError("gate_list is not given")

        self._params              = params           
        self._gate_list           = gate_list           
        self._qubit_op            = qubit_op        
        self._quantum_instance    = quantum_instance
        self._pre_circuit         = pre_circuit     
        self._post_circuit        = post_circuit     
        self._cost_type           = cost_type
        self.ref_state            = ref_state

        self._list_1gate = [idx for idx in range(len(self._gate_list)) if len(self._gate_list[idx])==1]
        self._list_2gate = [idx for idx in range(len(self._gate_list)) if len(self._gate_list[idx])==2]
        self._num_1gates = len(self._list_1gate)
        self._num_2gates = len(self._list_2gate)

        self._num_gates = self._num_1gates + self._num_2gates

        if params is None :
            self._params   = np.zeros(len(self._gate_list)*4).reshape(-1,4)
            self._u3params = np.zeros(len(self._gate_list)*4).reshape(-1,4)
            self._fixed_phase = np.zeros(len(self._gate_list))
        elif type(params) is list :
            self._params = np.array(params)
            self._fixed_phase = np.full( len(self._gate_list) ,np.pi/2)
        else:
            self._params = params
            self._fixed_phase = np.full( len(self._gate_list) ,np.pi/2)
        

        print("***** PQC ******") 
        print("   1-qubit gates list:",      self._list_1gate) 
        print("   Number of 1-qubit gates:", len(self._list_1gate)) 
        print("   2-qubit gates list ",      self._list_2gate) 
        print("   Number of 2-qubit gates:", len(self._list_2gate)) 
        print('   cost_type  : {}'.format(self._cost_type))

        @property
        def qubit_op(self) -> OperatorBase:
            return self._qubit_op

        @property
        def gate_list(self) -> np.ndarray:
            return self._gate_list

        @property
        def pre_circuit(self) -> QuantumCircuit:
            return self._pre_circuit

        @property
        def post_circuit(self) -> QuantumCircuit:
            return self._post_circuit



        if self._cost_type == "Fidelity" :
            if self.ref_state is None:
                self.ref_state = random_statevector(dims=2**self._num_qubits)

    def print_qubitOp(self):
        print("***** print qubit opreator ******")
        print(self._qubit_op)
        print(self._qubit_op.to_matrix_op())
        print("*********************************")


    def eval_ExactEigenSolver(self):
        exact_eigensolver = NumPyEigensolver(k=4)
        self._exact = exact_eigensolver.compute_eigenvalues(self._qubit_op)
        print("***** Exact EigenSolver ******")
        for n in range(4) :
            content  = '   ' + format(n, '2d')+ '-th eigenvalue=  '
            content += format(self._exact._eigenvalues[n].real, '.6f')
            print(content, flush=True)
        print("******************************")



    def print_u3unitary(self, u3param) :

        theta = u3param[0]
        phi   = u3param[1]
        lamb  = u3param[2]
        matrix  = np.arange(4, dtype='complex64').reshape(2,2)
        matrix[0][0] = np.cos(theta*0.5) 
        matrix[1][0] = np.sin(theta*0.5)*np.exp(phi*1j)
        matrix[0][1] =-np.sin(theta*0.5)*np.exp(lamb*1j)
        matrix[1][1] = np.cos(theta*0.5)*np.exp((phi+lamb)*1j)
        print("matrix {}\n".format(matrix))
        return matrix

    def cost_evaluation(self, params=None, display=False) -> float : 
        ### To construct FQS matrix, temporal paramters are casted to u3params and cu3params ####
        norm = None
        circuit = self.circuit_construct(params=params, display=display)

        if self._cost_type == "Expectation" :
            if self._quantum_instance.backend_name in {"aer_simulator", "qasm_simulator"}:
                post_rotation  =  PauliExpectation().convert(observable)
                sampler = CircuitSampler(self._quantum_instance).convert(post_rotation)
                output  =  sampler.eval().real
            elif self._quantum_instance.backend_name == "statevector_simulator" :
                observable  =  StateFn(self._qubit_op).adjoint() @ CircuitStateFn(circuit)
                output  =  observable.eval().real
            elif self._quantum_instance.backend_name == "aer_simulator_statevector" :
                observable  =  StateFn(self._qubit_op).adjoint() @ CircuitStateFn(circuit)
                sampler = CircuitSampler(backend=self._quantum_instance)
                output = sampler.convert( PauliExpectation().convert(observable) ).eval().real

        elif self._cost_type == "Fidelity" :
            if self._quantum_instance.backend_name is "statevector_simulator" :
                state    = Statevector(circuit)
                fidelity = state_fidelity(state, self.ref_state)
                output     = 1-fidelity
            if self._quantum_instance.backend_name in {"aer_simulator" or "qasm_simulator"}:
                raise TypeError("Fidelity with QASM is not supported yet")
        return output, norm 

    
    def circuit_construct(self, params=None, display =False) ->QuantumCircuit:
    
        if self._pre_circuit is None:
            qr = QuantumRegister(self._num_qubits)
            circuit = QuantumCircuit(qr)
        else:
            circuit = copy.deepcopy(self._pre_circuit)
            qr = circuit._qubits
        
        index=0
        """theta  = params[index]: phi = params[index+1]: lamb = params[index+2]"""
        for target, param, fixed_phase in zip(self._gate_list, params, self._fixed_phase) :
            if param[0] != 1 :
                u3param = np.zeros(4, dtype='float64') # theta, phi, lambda, phase
                u3param = self.param_QtoU3(param)
                circuit.u(u3param[0], u3param[1], u3param[2], target)

            index +=1
            
        if self._post_circuit is not None:
            circuit.append(self._post_circuit.to_instruction(), qr[0:self._num_qubits])
    
        if index != len(params):
            print("Index number does not much numer of u3 parameters! ")
            print("index= {0},  len(parameber)={1}".format(index, len(params)))
        
        if display : print(circuit)
        return circuit


    def _qubit_connecter(self, circuit, connect, param, fixed_phase) :
        """ old param """
        """ cindex: starting index in entangler list  
                    [[0,1],[1,2] ,,, ]  cidex=1 --> [1,2]
            cgate : an index target 
        """
        if len(param) != 4:
            print("connect=",connect)
            print("param=",param)
            raise TypeError('The array size of param should be 4, but not the case')
        if np.all(param == 0):
            pass
        elif np.all(param[0:3] - [np.pi,0,np.pi] == 0):
            circuit.cx(connect[0], connect[1])
            if fixed_phase+param[3] != np.pi :
                circuit.rz(fixed_phase + param[3] - np.pi, connect[0])

        elif np.all(param[0:3] - [0,0,np.pi] == 0):
            circuit.cz(connect[0], connect[1])
            if fixed_phase+param[3] != np.pi :
                circuit.rz(fixed_phase + param[3] - np.pi, connect[0])
        else :
            temporal_qc = QuantumCircuit(1)
            temporal_qc.u3(param[0],param[1],param[2], 0)
            custom = temporal_qc.to_gate().control(1)
            circuit.append(custom, [connect[0],connect[1]])
            if fixed_phase+param[3] != 0.0 :
                circuit.rz( fixed_phase + param[3], connect[0])
        return 

    def set_u3_angle(self, cparam) :
    
        cx = cparam[0]
        cy = cparam[1]
        cz = cparam[2]
        u3param = np.zeros(3, dtype='float64') # theta, phi, lambda
    
        if cz==0 : # theta ==0 
           u3param[0] = np.pi
           u3param[1] = math.atan2( cy,  cx) 
           u3param[2] = math.atan2( cy, -cx) 
        else :
           u3param[0] = 2*math.acos(cz)

           if u3param[0]!=0:
              cxx   = cx/math.sin(u3param[0]*0.5)
              cyy   = cy/math.sin(u3param[0]*0.5)
              u3param[1] =  math.atan2( cyy,  cxx) 
              u3param[2] =  math.atan2( cyy, -cxx) 
           else :
              u3param[1]  = np.pi
              u3param[2] = 0.0

        return u3param

    def u3toStatevector(self, params) -> Union[List[float], Dict[str, int]]:
        circuit = self.circuit_construct(params)
        return Statevector(circuit)

    def expressibility(self,
                       axis: Optional[str] = 'y',
                       random_type: Optional[str] = None,
                       output_name: Optional[str] = "expressibility.xvg",
                       max_iteration: Optional[int] = 1000
                       ):
        with open(output_name, 'a') as fp:
            for niter in range(max_iteration):
                u3params, __, __ = self._gen_u3params(random_type=random_type)
                sv1 = self.u3toStatevector(u3params)
                u3params, __, __ = self._gen_u3params(random_type=random_type)
                sv2 = self.u3toStatevector(u3params)
                fidelity = state_fidelity(sv1, sv2)
                content =' '
                content += format(niter, '4d')+ ' '
                content += format(fidelity, '.12f')+ ' '
                print(content, file=fp, flush=True)
        return 

    def set_1gate_param(self, method=None, axis='y', initial_type='random', params_1gate=None, target_list=None):

        ##### 1gate parameter should be described with quaternion ####
        if target_list is None :
            target_list = self._list_1gate
        ### setting self.params_1gate ###
        if params_1gate is None:
            if method.lower() in ["fqs","control-fqs", "controlled-fqs"]:
                self.set_param_fqs(target_list=target_list)
            elif method.lower() == "fraxis" :
                self.set_param_fraxis( target_list=target_list, initial_type='state-random')
            elif method.lower() in ["rotoselect", "rotosolve"]:
                self.set_param_roto(target_list=target_list, axis=axis, initial_type=initial_type)
        else :
            self.params_1gate = params_1gate
        return 


    def set_2gate_param(self, control_type='z', target_list=None):
        ##### 2gate parameter should be described with u3gate and phase gate ####

        if target_list is None :
            target_list = self._list_2gate

        ### setting self.params_2gate ###
        if control_type not in ["x","z", "I"]:
             print("control_type is", control_type)
             raise TypeError("either x, z, or random is allowed for entalger_type")

        if control_type.lower() == 'z':
            self.params_2gate=[[0, 0, np.pi,  np.pi/2]]*self._num_2gates
            for target in target_list :
                self._params[target]=[0, 0, np.pi,  np.pi/2]
                self._fixed_phase[target] = np.pi/2
        elif control_type.lower() == 'x':
            self.params_2gate=[[np.pi, 0, np.pi,  np.pi/2]]*self._num_2gates
            for target in target_list :
                self._params[target]=[np.pi, 0, np.pi,  np.pi/2]
                self._fixed_phase[target] = np.pi/2
        elif control_type.lower() == 'I':
            self.params_2gate=[[0, 0, 0, 0]]*self._num_2gates
            for target in target_list :
                self._params[target]=[0, 0, 0, 0]
        return 

    def param_mixer(self) : 
        self._params = []
        index1=0
        index2=0
        for gate in self._gate_list :
            if len(gate) == 1:
                self._params.append(self.params_1gate[index1])
                index1 += 1
            if len(gate) == 2:
                self._params.append(self.params_2gate[index2])
                index2 += 1

    def set_param_fqs(self, target_list) :
        self.params_1gate = []
        for target in target_list :
            quaternion       = self._gen_unit_quaternion()
            self._params[target] = quaternion

    def set_param_fraxis(self, target_list=None, initial_type='state-random') :

        for target in target_list :
            if  initial_type.lower() == "parameter-random":
                nxyz = self.gen_cparam_ParamRandom()
            elif initial_type.lower() == "state-random":
                nxyz = self.gen_cparam_StateRandom()
            elif initial_type.lower() == "perturb":
                nxyz = self.gen_cparam_perturb()
            quaternion = [np.cos(np.pi/2), np.sin(np.pi/2)*nxyz[0], np.sin(np.pi/2)*nxyz[1], np.sin(np.pi/2)*nxyz[2]] 
            self._params[target] = quaternion
        return

    def set_param_roto(self, target_list=None, initial_type='random', axis='y') :
        for target in target_list :
            self._params[target] = self.gen_param_roto(initial_type=initial_type, axis=axis)
        return

    def gen_param_roto(self, initial_type='random', axis='y') :

        if   initial_type.lower() == "random":
            angle = (np.random.rand()*2.0-1)*math.pi
        elif initial_type.lower() == 'perturb':
            angle = (np.random.rand()*2.0-1)*math.pi*0.01
        else:
            raise TypeError("GateParameter.gen_rotational_angle: initial_type is either random, or perturb")
        nxyz  = self.gen_cxyz(axis=axis)
        quaternion = [np.cos(angle/2), np.sin(angle/2)*nxyz[0], np.sin(angle/2)*nxyz[1], np.sin(angle/2)*nxyz[2]] 
        return quaternion

    def gen_cxyz(self, axis: Optional[str]= None) : 
        if axis == "random" :
            cparams = [0,0,0]
            cparams[np.random.randint(0,3)] = 1
        elif axis in ['x','y','z']:
            cparams = [0,0,0]
            if axis == 'x' :
               cparams[0] = 1
            elif axis == 'y' :
               cparams[1] = 1
            elif axis == 'z' :
               cparams[2] = 1
        else :
            raise TypeError("GateParameter.gen_fraxis_param; initial_type is either x, y, z or random")
        return cparams

    def gen_u3params(self, random_type: Optional[str] = "state-random",):
        cparams  = self.gen_fraxis_cparam(random_type=random_type)
        u3params = self.paramset_CtoU(cparams, self._angle, self._num_gates)
        return u3params, cparams, self._angle

    def set_refstate(self) :
        self.ref_state = random_statevector(dims=2**self._num_qubits)

    def _gen_unit_quaternion(self) -> List[float] :
        r   = np.random.rand()
        phi1 = np.random.rand()*2*np.pi
        phi2 = np.random.rand()*2*np.pi
        q = [np.sqrt(r)*np.sin(phi1), np.sqrt(r)*np.cos(phi1), np.sqrt(1-r)*np.sin(phi2), np.sqrt(1-r)*np.cos(phi2)]
        return q


