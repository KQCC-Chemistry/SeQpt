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
from qiskit.opflow.operator_base import OperatorBase
from qiskit.utils import QuantumInstance
from qiskit.quantum_info.states.quantum_state import QuantumState

from . import PlasticPQC 
import math 
import cmath 
import pprint

import sys


class RotationOf():

    def rotosolve(self, idx=None, current_val=None):

        private_params = copy.deepcopy(self._pqc._params)
        if current_val is None:
            current_val, _   = self._pqc.cost_evaluation(params=private_params)

        axis = np.zeros(3, dtype='float64')          # nx, ny, nz
        axis = self._pqc._params[idx][1:4]
        angle = math.atan2(np.linalg.norm(axis), private_params[idx][0])*2
        angle = self._pqc.parameter_period(angle, [0,2*np.pi])


        if np.all(axis==0.0) :
            raise ValueError("Rotation axis in Rotosolve is ambiguous!!")
        if np.all(math.sin(angle/2) !=0.0 ) :
            axis = axis/math.sin(angle/2)

        private_params[idx][0] = math.cos((angle+np.pi*0.5)/2)
        scale = math.sin((angle+np.pi*0.5)/2)  
        private_params[idx][1:4] =  axis*scale
        Mpls,_  = self._pqc.cost_evaluation(private_params)

        private_params[idx][0] = math.cos((angle-np.pi*0.5)/2)
        scale = math.sin((angle-np.pi*0.5)/2)  
        private_params[idx][1:4] = axis*scale
        Mmns,_  = self._pqc.cost_evaluation(private_params)
        
        y = 2*current_val - Mpls - Mmns
        x = Mpls - Mmns
        A = 0.5*np.sqrt(y**2 + x**2) 
        B = math.atan2(y,x) - angle
        C = 0.5*(Mpls+Mmns) 
        expected_val = C-A

        opt_angle = -B -np.pi/2
        self._pqc._params[idx][0] = math.cos(opt_angle/2)
        scale = math.sin(opt_angle/2)  
        if scale !=0 :
            self._pqc._params[idx][1:4] = axis*scale

#        check, _ = self._pqc.cost_evaluation(params=self._pqc._params)
#        if abs(check-expected_val > 1e-2) :
#            print("Warning expect_val=",expected_val, ",  measured=",check)
#            print("   angle(degree): % 3.1f -> % 3.1f  " % (angle/np.pi*180, opt_angle/np.pi*180), flush=True)
#            print("   axis=",axis, "  -> new_axis=", self._pqc._params[idx][1:4])
#            print("   param=",self._pqc._params[idx])
#            sys.exit(1)

        print("   angle(degree): % 3.1f -> % 3.1f  " % (angle/np.pi*180, opt_angle/np.pi*180), flush=True)
        content  ='   Improved = '
        content += format(current_val-expected_val, '4.1e')
        print(content, flush=True)
        return expected_val


    def rotoselect(self, idx=None, current_val=None):

        axis  = None
        Axes = {'x': [1,0,0], 'y':[0,1,0], 'z':[0,0,1]}

        if current_val is None:
            current_val, _   = self._pqc.cost_evaluation(params=self._pqc._params)
 
        private_params      = copy.deepcopy(self._pqc._params)
        private_params[idx] = self._pqc.param_CtoQ(Axes['x'],  0)
        M0, _   = self._pqc.cost_evaluation(params=private_params)


        for direct in Axes :
            private_params[idx] = self._pqc.param_CtoQ(Axes[direct],  np.pi*0.5)
            Mpls, _  = self._pqc.cost_evaluation(params=private_params)
            private_params[idx] = self._pqc.param_CtoQ(Axes[direct], -np.pi*0.5)
            Mmns, _  = self._pqc.cost_evaluation(params=private_params)
            y = 2*M0 - Mpls - Mmns
            x = Mpls - Mmns
            A = 0.5*np.sqrt(y**2 + x**2) 
            B = math.atan2(y,x)
            C = 0.5*(Mpls+Mmns) 
            tmp_angle = -np.pi*0.5 - B 
            expected_val   = C-A
            if current_val >= expected_val :
                axis        = direct
                current_val = expected_val
                angle       = tmp_angle
                private_params[idx] = self._pqc.param_CtoQ(Axes[axis], angle)
            
        if axis is not None :
            print('   new_angle     : {}'.format(angle), flush=True)
#            self._pqc._angle[idx]  = angle
#            print('   cparams=', self._pqc._cparams[idx],'->',Axes[axis], flush=True)
#            self._pqc._cparams[idx]  = Axes[axis]
            self._pqc._params[idx] = self._pqc.param_CtoQ(Axes[axis], angle)
        else :
            print('   params is not Updated', flush=True)

        return current_val


