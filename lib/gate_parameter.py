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
from qiskit.circuit.library import RealAmplitudes
#from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.expectations import ExpectationFactory
from qiskit.opflow.expectations.expectation_base import ExpectationBase
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.converters import CircuitSampler
from qiskit.tools.visualization import circuit_drawer

import math 
import cmath 
import pprint

logger = logging.getLogger(__name__)

class GateParameter:

    def __init__(self,
                 num_qubits      : int = None, 
                 depth           : int = None, 
                 params          : List[float] =None, 
                 ) -> None:


        self._num_qubits          = num_qubits        
        self._depth               = depth        
        self._params              = params


        @property
        def num_qubits(self) -> int:
            return self._num_quibits

        @property
        def depth(self) -> int:
            return self._depth

        @property
        def u3params(self) -> np.ndarray:
            return self._u3params

        @u3params.setter
        def u3params(self, u3params) :
            self._u3params = array(u3params).reshape(-1,3)

        @property
        def cparams(self) -> np.ndarray:
            return self._cparams

        @cparams.setter
        def cparams(self, cparams) -> np.ndarray:
            self._cparams = array(cparams).reshape(-1,3)

        @property
        def angle(self) -> np.ndarray:
            return self._angle

        @angle.setter
        def angle(self, angle) -> np.ndarray:
            self._angle = angle



    def gen_cparam_perturb(self, num_gates) :
        cparams=[]
        for i in range(num_gates) :
            pset=[]
            for param in range(3) :
                if param%3 == 2 :
                    rr = (np.random.rand()*2.0-1)*1e-1 + 1.0
                    pset = np.append(pset, rr)
                else :
                    rr = (np.random.rand()*2.0-1)*1e-1
                    pset = np.append(pset, rr)
            pset = pset/np.linalg.norm(pset)
            cparams = np.append(cparams, pset)
        return cparams.reshape(-1,3)


    def gen_cparam_ParamRandom(self, num_gates) :
        cparams=[]
        probe=[0,0,1]
        for niter in range(num_gates) :
            pset    = np.random.rand(3)            #range[0,1]
            phi     = np.random.rand()*np.pi       #range[0,pi]
            psi     = (np.random.rand()*2-1)*np.pi #range[-pi,pi]
            pset[0] =  math.sin(phi)*math.cos(psi)
            pset[1] =  math.sin(phi)*math.sin(psi)
            pset[2] =  math.cos(phi)
            cparams = np.append(cparams, pset)
        return cparams.reshape(-1,3)

    def gen_cparam_StateRandom(self) :
        cparam  = np.random.normal(0, 1, 3)
        cparam  = cparam/np.linalg.norm(cparam)
        return cparam


    def param_CtoU(self, cparam, angle) :
        cx = cparam[0]
        cy = cparam[1]
        cz = cparam[2]
        u3param = np.zeros(3, dtype='float64')
        matrix  = np.arange(4, dtype='complex64').reshape(2,2)
        matrix[0][0] = np.cos(angle*0.5) - np.sin(angle*0.5)*cz*1j
        matrix[1][0] =                   - np.sin(angle*0.5)*(cx+cy*1j)*1j
        matrix[0][1] =                   - np.sin(angle*0.5)*(cx-cy*1j)*1j
        matrix[1][1] = np.cos(angle*0.5) + np.sin(angle*0.5)*cz*1j

        if abs(matrix[0][0]) != 0 :
            gphase = matrix[0][0]/abs(matrix[0][0])
            matrix /= gphase

        if matrix[1][0].real >= 0 : 
           u3param[0] = math.atan2( abs(matrix[1][0]), matrix[0][0].real)*2     #  phi+lambda
        else :
           u3param[0] = math.atan2(-abs(matrix[1][0]), matrix[0][0].real)*2     #  phi+lambda

        if u3param[0] != 0 : 
           u3param[1] = cmath.phase( matrix[1][0]/ cmath.sin(u3param[0]) ) 
           u3param[2] = cmath.phase(-matrix[0][1]/ cmath.sin(u3param[0]) ) 
        else:
           u3param[1] = cmath.phase( matrix[1][1]) 
           u3param[2] = 0.0
        return u3param

    def param_CtoQ(self, cparam, angle) :
        u3param = self.param_CtoU(cparam=cparam, angle=angle) 
        quaternion = self.param_U3toQ(u3param=u3param) 
        return quaternion

    def param_U3toQ(self, u3param=None) :
        quaternion = np.zeros(4, dtype='float64')
        matrix  = np.arange(4, dtype='complex64').reshape(2,2)

        matrix[0][0] = np.cos(u3param[0]*0.5)*np.exp((-u3param[1]-u3param[2])*0.5*1j)
        matrix[0][1] =-np.sin(u3param[0]*0.5)*np.exp((-u3param[1]+u3param[2])*0.5*1j)
        matrix[1][0] = np.sin(u3param[0]*0.5)*np.exp(( u3param[1]-u3param[2])*0.5*1j)
        matrix[1][1] = np.cos(u3param[0]*0.5)*np.exp(( u3param[1]+u3param[2])*0.5*1j)

        A   = matrix[1][1] + matrix[0][0]
        B   = matrix[1][1] - matrix[0][0]              # cos(theta/2)
        C   = matrix[1][0] + matrix[0][1]
        D   = matrix[1][0] - matrix[0][1]

        quaternion = [A.real/2, C.imag/2, D.real/2, B.imag/2]

        return quaternion


    def paramset_CtoU(self, cparams, angle, num_gates) :
        index = 0
        gate     = 0
        u3params  = []
        for gate in range(num_gates) :
            pset     = cparams[index]
            u3params = np.append(u3params, self.param_CtoU(pset, angle[gate]))
            index += 1
        return u3params.reshape(-1,3)


    def param_UtoC(self, u3param) :
        phi   = u3param[0]
        psi   = u3param[1]
        lamb  = u3param[2]
        matrix[0][0] =+np.cos(phi/2) 
        matrix[1][0] =+cmath.phase(psi *1j)* np.sin(phi/2)
        matrix[0][1] =-cmath.phase(lamb*1j)* np.sin(phi/2)
        matrix[1][1] =+cmath.phase((psi+lamb)*1j)*np.cos(phi/2) 
        return u3param

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
    
    def parameter_period(self, angle, period: [np.ndarray]) :
    
       leng = period[1] - period[0]
       while angle < period[0] or angle > period[1]:
           if angle > period[1]:
               angle -= leng
           elif angle < period[0] :
               angle += leng
       return angle

    def parameters_period(self,  parameters: [np.ndarray], period: [np.ndarray]) :
    
       for index in range(len(parameters)):
          parameters[index] = self.parameter_period(parameters[index], period)
       return parameters

    def param_QtoExC(self, quaternion):
        # quaternion[0]: cos(theta/2)    # quaternion[1]: sin(theta/2)*nx
        # quaternion[2]: sin(theta/2)*ny # quaternion[3]: sin(theta/2)*nz

        # if quaternion[0] < 0:
        #     quaternion=-1*quaternion
        axis = np.array([quaternion[1], quaternion[2], quaternion[3]])

        # norm: abs(sin(theta/2))
        norm = np.linalg.norm(axis)

        if norm != 0:
            angle = 2*np.arctan2(norm, quaternion[0])
            # normalization of nx, ny, nz
            excparam = angle*axis/(2*norm)
        else:
            excparam = axis

        # excparam: n_x,n_y,n_z * (angle/2)
        #                           why?
        return excparam

    def param_QtoC(self, four_vector):
        # quaternion[0]: cos(theta/2)
        # quaternion[1]: sin(theta/2)*nx
        # quaternion[2]: sin(theta/2)*ny
        # quaternion[3]: sin(theta/2)*nz

        axis = np.array([four_vector[1], four_vector[2], four_vector[3]])
        # norm: abs(sin(theta/2))
        norm = np.linalg.norm(axis)

        if norm != 0:
            psi = 2*np.arctan2(norm,four_vector[0])
            cparam = axis/norm
        else:
            cparam = np.array(0,0,1)
        return cparam, angle


#   there is buts in this method  #    
    def param_QtoU3_new(self, Q) -> List[float]:
#        excparam  = self.param_QtoExC(four_vector)
#        u3 = self.param_ExCtoU3(excparam)

        # U[0]: theta,    U[1]: phi
        # U[2]: lambda,   U[3]: global phase
        # Q[0]: cos(theta/2),    Q[1]: sin(theta/2)*nx
        # Q[2]: sin(theta/2)*ny, Q[3]: sin(theta/2)*nz

        u3param = np.zeros(4, dtype='float64')  # theta, phi, lambda
        if np.all(Q == 0.0):
            pass
        else:
            sign = np.sign(Q[0])
            cos2, u3param[3] = cmath.polar(Q[0]-Q[3]*1j)   # u3param[3] = [-pi, pi]
            u3param[3]=u3param[3]+np.pi                    # u3param[3] = [0, 2*pi]

            if Q[0]>=0 and Q[3]>=0 :
                pass
            elif Q[0]<0 and Q[3]>=0 :
                cos2=-cos2
                u3param[3]= np.pi - u3param[3]                 # pi/2-(u3param[3]-pi/2); u3param[3] = [  0, 2pi]
            elif Q[0]<0 and Q[3]<0 :
                cos2=-cos2
                u3param[3]= -np.pi - u3param[3]                # u3param[3] = [  0, 2pi]
            elif Q[0]>=0 and Q[3]<0 :
                u3param[3]= u3param[3]+2*np.pi                 # u3param[3] = [  0, 2pi]
                
            sign = np.sign(Q[2])
            sin2a, phi   = cmath.polar((-Q[1]*1j+Q[2]))
            sin2b, lamb = cmath.polar(( Q[1]*1j+Q[2]))
            
            print("Q=",Q)
            print("phi=",phi, "   lambda=", lamb,  "  u3param[3]=", u3param[3])

            if sin2a==sin2b :
                sin2=sin2a
            radius3, angle3 = cmath.polar((+Q[1]*1j+Q[2])*sign)
            print("cos2=",cos2, "sin2=", sin2)
            u3param[0] = math.atan2(sin2, cos2)*2
            u3param[1] = phi 
            u3param[2] = lamb

#### for debug ##
#        mat1 = np.arange(4, dtype='complex64').reshape(2,2)        
#        mat1[0][0]= Q[0]-Q[3]*1j
#        mat1[1][1]= Q[0]+Q[3]*1j
#        mat1[0][1]=-Q[2]-Q[1]*1j
#        mat1[1][0]= Q[2]-Q[1]*1j
#        
#        mat2 = np.arange(4, dtype='complex64').reshape(2,2)        
#        mat2[0][0]= math.cos(u3param[0]/2)
#        mat2[1][1]= math.cos(u3param[0]/2)*cmath.exp(1j*(u3param[1]+u3param[2]))
#        mat2[0][1]=-math.sin(u3param[0]/2)*cmath.exp(1j*u3param[2])
#        mat2[1][0]= math.sin(u3param[0]/2)*cmath.exp(1j*u3param[1])
#        print("mat1=", mat1)
#        print("mat2=", mat2)
#        print("\n")

        return u3param

#   there is buts in this method  #    
    def param_QtoU3(self, Q) -> List[float]:
#        excparam  = self.param_QtoExC(four_vector)
#        u3 = self.param_ExCtoU3(excparam)

        # U[0]: theta,    U[1]: phi
        # U[2]: lambda,   U[3]: global phase
        # Q[0]: cos(theta/2),    Q[1]: sin(theta/2)*nx
        # Q[2]: sin(theta/2)*ny, Q[3]: sin(theta/2)*nz

        angle = math.acos(Q[0])*2   # angle = [0, 2*pi]
        u3param = np.zeros(4, dtype='float64')  # theta, phi, lambda

        if math.sin(angle/2) == 0.0:
            pass
        else:
            nx = Q[1]/(math.sin(angle*0.5))
            ny = Q[2]/(math.sin(angle*0.5))
            nz = Q[3]/(math.sin(angle*0.5))

            matrix  = np.arange(4, dtype='complex64').reshape(2,2)
            matrix[0][0] = np.cos(angle*0.5) - np.sin(angle*0.5)*nz*1j
            matrix[1][0] =                   - np.sin(angle*0.5)*(nx+ny*1j)*1j
            matrix[0][1] =                   - np.sin(angle*0.5)*(nx-ny*1j)*1j
            matrix[1][1] = np.cos(angle*0.5) + np.sin(angle*0.5)*nz*1j

            if matrix[0][0] == 0 :
                u3param[3] = 0
                u3param[0] = np.pi
                u3param[2] = 0.5*(cmath.phase(matrix[1][1])+cmath.phase(matrix[1][0]/matrix[0][1])-u3param[3]-np.pi)
                u3param[1] = cmath.phase(matrix[1][1])-u3param[3]-u3param[2]
            else :
                radius, u3param[3] = cmath.polar(matrix[0][0])
                matrix /= cmath.rect(1,u3param[3])
                if matrix[0][1] == 0 :
                    u3param[0] = 0.0
                    u3param[1] = cmath.phase(matrix[1][0])
                    u3param[2] = cmath.phase(matrix[1][1])-u3param[1]
                else:
                    pls        = cmath.phase(matrix[1][1]/matrix[0][0])
                    mns        = cmath.phase(-matrix[1][0]/matrix[0][1])
                    u3param[1] = (pls+mns)*0.5
                    u3param[2] = (pls-mns)*0.5

                    ytheta     = matrix[1][0]/cmath.exp(u3param[1]*1j)
                    u3param[0] = self.parameter_period(2*math.atan2(ytheta.real, matrix[0][0].real), [-np.pi,np.pi])
#                    u3param[0] = self.parameter_period(2*math.atan2(ytheta.real, matrix[0][0].real), [-np.pi,np.pi])

#                    u3param[0] = self.parameter_period(2*math.atan2(ytheta.real, matrix[0][0].real), [0, 2*np.pi])
#                    u3param[1] = self.parameter_period(u3param[1], [0, 2*np.pi])
#                    u3param[2] = self.parameter_period(u3param[2], [0, 2*np.pi])
#                    u3param[3] = self.parameter_period(u3param[3], [0, 2*np.pi])

        # return cuparam
#        print(matrix)

#        ### debug
#        mat1 = np.arange(4, dtype='complex64').reshape(2,2)        
#        mat1[0][0]= Q[0]-Q[3]*1j
#        mat1[1][1]= Q[0]+Q[3]*1j
#        mat1[0][1]=-Q[2]-Q[1]*1j
#        mat1[1][0]= Q[2]-Q[1]*1j
#        
#        mat2 = np.arange(4, dtype='complex64').reshape(2,2)        
#        mat2[0][0]= math.cos(u3param[0]/2)
#        mat2[1][1]= math.cos(u3param[0]/2)*cmath.exp(1j*(u3param[1]+u3param[2]))
#        mat2[0][1]=-math.sin(u3param[0]/2)*cmath.exp(1j*u3param[2])
#        mat2[1][0]= math.sin(u3param[0]/2)*cmath.exp(1j*u3param[1])
#        print("mat1=", mat1)
#        print("mat2=", mat2)
#        print("\n")
        return u3param

    def param_QtoU4(self, Q) -> List[float]:
        # Q[0]: cos(theta/2),    Q[1]: sin(theta/2)*nx
        # Q[2]: sin(theta/2)*ny, Q[3]: sin(theta/2)*nz

        u3param = np.zeros(4, dtype='float64')
        if Q[0]==1 :
            return [0,0,0,0]

        if Q[0]==0 and Q[3]==0 :
            u3param[0] = np.pi
            mns = math.atan2(-Q[1],Q[2])      # mns = (phi-lambda)/2 
            if mns < 0 : 
                mns = mns + np.pi
            pls = 0                       # (phi-lambda)/2 does not affect the expectation but global phase. For pls=0, global phase=0

        elif Q[1]==0 and Q[2]==0 :
            u3param[0] = 0
            pls = math.atan2( Q[3], Q[0]) # (phi+lambda)/2
            mns = pls
        else:
            pls = math.atan2( Q[3], Q[0]) # (phi+lambda)/2
            mns = math.atan2(-Q[1], Q[2]) # (phi-lambda)/2 
            cos = np.sqrt(Q[0]**2+Q[3]**2)
            sin = np.sqrt(Q[1]**2+Q[2]**2)
            u3param[0] = math.atan2(sin, cos)*2

        u3param[1] = (pls+mns)
        u3param[2] = (pls-mns)
        u3param[3]=pls/2
        return u3param


    def param_ExCtoU3(self, excparam) -> List[float]:
        # excparam[0]: cos(theta/2),    excparam[1]: sin(theta/2)*nx
        # excparam[2]: sin(theta/2)*ny, excparam[3]: sin(theta/2)*nz

        # norm: abs(sin(theta/2))
        angle = 2*np.linalg.norm(excparam, 2)

        # theta, phi, lambda, gamma
        # gamma represents a global phase
        cuparam = np.zeros(4, dtype='float64')

        if angle == 0.0:
            u3param = np.zeros(3, dtype='float64') # theta, phi, lambda
            cuparam = np.zeros(4, dtype='float64') # theta, phi, lambda
        else:
            cx = excparam[0]/(angle*0.5)
            cy = excparam[1]/(angle*0.5)
            cz = excparam[2]/(angle*0.5)

            matrix  = np.arange(4, dtype='complex64').reshape(2,2)
            matrix[0][0] = np.cos(angle*0.5) - np.sin(angle*0.5)*cz*1j
            matrix[1][0] =                   - np.sin(angle*0.5)*(cx+cy*1j)*1j
            matrix[0][1] =                   - np.sin(angle*0.5)*(cx-cy*1j)*1j
            matrix[1][1] = np.cos(angle*0.5) + np.sin(angle*0.5)*cz*1j

            if matrix[0][0] == 0 :
                cuparam[3] = 0
                cuparam[0] = np.pi
                cuparam[2] = 0.5*(cmath.phase(matrix[1][1])+cmath.phase(matrix[1][0]/matrix[0][1])-cuparam[3]-np.pi)
                cuparam[1] = cmath.phase(matrix[1][1])-cuparam[3]-cuparam[2]
            else :
                radius, cuparam[3] = cmath.polar(matrix[0][0])
                matrix /= cmath.rect(1,cuparam[3])
                if matrix[0][1] == 0 :
                    cuparam[0] = 0.0
                    cuparam[1] = cmath.phase(matrix[1][0])
                    cuparam[2] = cmath.phase(matrix[1][1])-cuparam[1]
                else:
                    pls        = cmath.phase(matrix[1][1]/matrix[0][0])
                    mns        = cmath.phase(-matrix[1][0]/matrix[0][1])
                    cuparam[1] = (pls+mns)*0.5
                    cuparam[2] = (pls-mns)*0.5

                    ytheta     = matrix[1][0]/cmath.exp(cuparam[1]*1j)
                    cuparam[0] = self.parameter_period(2*math.atan2(ytheta.real, matrix[0][0].real), [-np.pi,np.pi])

            ## u3 params
#            u3param = np.array([cuparam[0], cuparam[1], cuparam[2]])
            u3param = [cuparam[0], cuparam[1], cuparam[2], cuparam[3]]

        # return cuparam
        return u3param


