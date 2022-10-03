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
from  . import PlasticPQC
#from .pqc import PQC
from .roto import RotationOf
from .fraxis import FreeAxisSelection
from .fqs import FreeQuaternionSelection

import math
import cmath
import pprint


class SequentialOptimizer(RotationOf, FreeAxisSelection, FreeQuaternionSelection):

    def __init__(self,
                 pqc             : Union[PlasticPQC]=None,
                 threshold       : Optional[float] = 1e-8,
                 output_name     : Optional[str] = "traj.xvg"
                 ) -> None:

        if isinstance(pqc, PlasticPQC) :
            self._pqc = pqc
        else:
            raise QiskitError("pqc is not in class PQC")

        self.threshold     = threshold
        self.output_name   = output_name
        self._eval_count  = 0
            

    def search_minimum(self, method=None, current_val=None, target_list=None, max_iteration=1):
        print('\n***********************************', flush=True)
        print('Start VQE with Squentioal Optimizer', flush=True)
        print('>>> method=', method, flush=True)

        ## initial params
        initial_params = copy.deepcopy(self._pqc._params)

        ## initial energy
        if current_val is None :
            initial_val, _   = self._pqc.cost_evaluation(params=initial_params, display=True)
            self.output_result(initial_val, initial_params, self.output_name)
        else :
            initial_val, _   = current_val

        if target_list is None  :
            target_list = self.set_target(method=method)

        current_val = initial_val
        for cycle in range(max_iteration):
            if cycle == 0:
                previous_val = current_val
            elif np.abs(previous_val - current_val) < self.threshold  and cycle > 0:
                print('\n\nDelta={}'.format(abs(current_val - previous_val)), flush=True)
                print('convegence at {} interation'.format(cycle), flush=True)
                break
            else:
                print('End of update cycle:  Delta ={0}'.format(current_val - previous_val), flush=True)
                previous_val = current_val
            print('\n ***** Iteration cycle {} *****'.format(cycle), flush=True)
            ## optimize params
            current_val =  self.sweep(method=method, target_list=target_list, cycle=cycle, current_val=current_val)
            if cycle == max_iteration-1:
                print('\n <<<<<<< Reached to maxiteration {}  >>>>>>\n'.format(max_iteration), flush=True)

        final_val, _ = self._pqc.cost_evaluation(params=self._pqc._params, display=True)

        print('>>> End of search_minimum with', method, "<<<", flush=True)
        print('   initial value    : % .6f' % initial_val)
        print('   final value      : % .6f' % final_val )
        print('   **** initial & final params :')
        for target in target_list :
            print("   index=",target, "   ",initial_params[target], " ----> ", self._pqc._params[target])
        return final_val

    def sweep(self, method, target_list=None, cycle=None, current_val=None):
        if target_list is None  :
            target_list = self.set_target(method=method)

        if current_val is None :
            current_val, _   = self._pqc.cost_evaluation(params=self._pqc._params, display=True )

        print("target_list=",target_list)
        for idx in target_list :
            if cycle is None :
                print('\n\n idx={0}'.format(idx), flush=True)
            else :
                print('\n##### iter=%d, idx=%2d, method=%s, value=% 9.5f #####'% (cycle, idx, method, current_val), flush=True)
            current_val = self.update(method=method, idx=idx, current_val=current_val)
            print("   current energy = % 8.6f " %  current_val, flush=True)
#            self.output_result(current_val, self._pqc._u3params, self.output_name)
        return current_val

    def entangle_adder(self, current_val=None, max_trial=10):

        if current_val is None:
            current_val, _ = self._pqc.cost_evaluation(params=self._pqc._params)

        num_space = len(self._pqc._entangler)+1
        initial_entangler = copy.deepcopy(self._pqc._entangler)
        initial_cu3params = copy.deepcopy(self._pqc._cu3params)

        update_flag = False
        space_list=[]
        qubit_pair=[]
        value_list=[]
        param_list=np.array([])
        for trial in range(max_trial) :
            print("   \n>>> Append Traial =",trial, flush=True )
            insert_space  = np.random.randint(0,num_space)
            control_qubit = np.random.randint(0,self._pqc._num_qubits)
            target_qubit  = (control_qubit+np.random.randint(1,self._pqc._num_qubits))%self._pqc._num_qubits
            self._pqc._entangler = copy.deepcopy(initial_entangler)
            self._pqc._entangler.insert(insert_space, [control_qubit, target_qubit])
            self._pqc._cu3params = np.insert(initial_cu3params, insert_space, np.array([0,0,0,0]),axis=0).reshape(-1,4)

            print("    entangler    =", self._pqc._entangler, flush=True)
            tmp_value, tmp_param = self.controlled_fqs(gate_idx=insert_space, current_val=current_val, cu3params=self._pqc._cu3params)
            value_list.append(tmp_value)  
            param_list= np.append(param_list, tmp_param[insert_space]).reshape(-1,4)  
            space_list.append(insert_space)
            qubit_pair.append([control_qubit, target_qubit])

        if np.amin(value_list) < current_val :
            sid = np.argmin(value_list)
            current_val = np.amin(value_list)
            self._pqc._entangler = copy.deepcopy(initial_entangler)
            self._pqc._entangler.insert(space_list[sid], qubit_pair[sid])
            self._pqc._cu3params = np.insert(initial_cu3params, space_list[sid], param_list[sid], axis=0).reshape(-1,4)
            self._num_cgates = len(self._pqc._cu3params)
            update_flag = True

        if update_flag is True:
            print("**** Successfully appending entangler ****")
            print("    current_val  =",current_val, flush=True)
            print("    entangler    =",self._pqc._entangler, flush=True)
            print("    cu3params    =",self._pqc._cu3params, flush=True)
        else :
            self._pqc._entangler = copy.deepcopy(initial_entangler)
            self._pqc._cu3params = copy.deepcopy(initial_cu3params)

        self.output_result(current_val, self._pqc._u3params, self.output_name)
        return current_val

    def entangle_remover(self, current_val=None, threshold=1e-2, max_entangler=8):

        remove_flag = False

        print("***************************")
        print("***** entangle_remover ****")
        print("***************************")

        for gate_id in reversed(range(len(self._pqc._entangler))) :
            mid_measure = MidCircuitMeasure(cgate_id=gate_id, is_measure={'c':'z'}, basis=[0], replace_param=np.array([[1,0,0,0],[1,0,0,0]]) )
            _, norm_cz0 = self._pqc.cost_evaluation(u3params=self._pqc._u3params, cu3params=self._pqc._cu3params, mid_measure=mid_measure)
            if 1-norm_cz0**2 < threshold :
                print("      ",gate_id,"-th gate was removed!")
                del self._pqc._entangler[gate_id]
                self._pqc._cu3params = np.delete(self._pqc._cu3params, gate_id, 0)
                print("   previous value =",current_val)
                current_val, _ = self._pqc.cost_evaluation(u3params=self._pqc._u3params, cu3params=self._pqc._cu3params)
                print("   new value      =",current_val)
                remove_flag = True

        if remove_flag is False and  len(self._pqc._entangler) > max_entangler :
            print("   >>>>> No small L0 value ")
            initial_cu3params = copy.deepcopy(self._pqc._cu3params)
            cost_list = []
            for gate_id in range(len(self._pqc._entangler)) :
                self._pqc._cu3params = copy.deepcopy(initial_cu3params)
                self._pqc._cu3params[gate_id] = np.array([0,0,0,0])
                temp_val, _ = self._pqc.cost_evaluation(u3params=self._pqc._u3params, cu3params=self._pqc._cu3params)
                cost_list = np.append(cost_list, temp_val)

            self._pqc._cu3params = copy.deepcopy(initial_cu3params)
            min_id = np.argmin(cost_list)
            print("   cost_list=",cost_list)
            print("   previous value =",current_val)
            current_val = cost_list[min_id]
            print("   current value  =",current_val)
            del self._pqc._entangler[min_id]
            print("   New entangler list", self._pqc._entangler)
            self._pqc._cu3params = np.delete(self._pqc._cu3params, min_id, 0)

        self._num_cgates = len(self._pqc._cu3params)
        self.output_result(current_val, self._pqc._u3params, self.output_name)

        return current_val

    def set_target(self, method):
        if method.lower() in [ "controlled-fqs", "control-fqs", "controlfqs"] :
            target_num  = self._pqc._num_2gates
            target_list = self._pqc._list_2gate
        elif method.lower() in [ "fqs","fraxis","rotoselect","rotosolve","nft" ] :
            target_num  = self._pqc._num_1gates
            target_list = self._pqc._list_1gate
        else :
            raise TypeError("Method name was not properly provided")
        return target_list


    def update(self, method, idx, current_val=None):
        if method.lower() in [ "controlled-fqs", "control-fqs", "controlfqs"] :
            current_val, self._pqc._params = self.controlled_fqs(gate_idx=idx, current_val=current_val)
        if method.lower() == "fqs" :
            current_val = self.free_quaternion_selection(idx, current_val)
        elif method.lower() == "fraxis" :
            current_val = self.free_axis_selection(idx=idx, current_val=current_val)
        elif method.lower() == "rotoselect" :
            current_val = self.rotoselect(idx, current_val=current_val)
        elif method.lower() == "rotosolve" :
            current_val = self.rotosolve(idx, current_val=current_val)

        self.output_result(current_val, self._pqc._params, self.output_name)
        return current_val


    def output_result(self, val, params, output_name):
        param_list = params.tolist()
#        param_list = params
        with open(output_name, 'a') as fp:
            content =' '
            content += format(self._eval_count, '4d')+ ' '
            content += format(val, '.12f')+ ' '
            content += "{}".format(param_list, '.8f')

            print(content, file=fp, flush=True)
            self._eval_count += 1


