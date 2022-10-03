#!/usr/bin/env python
from typing import Optional, List
import math
import numpy as np
import sys  


def CircuitDesigner(num_qubits      : int = None,
                    num_layer       : int = None,
                    entangler_type  : Optional[str]  = "cyclic",
                    double          : bool=False 
                    ) :

    gate_list=[]
    num_gates=0
    target_list = []
    if entangler_type.lower() == "cascading" :
        for n in range(num_qubits):
            gate_list.append([n])
            num_gates = num_gates+1

        for d in range(num_layer):
            for n in range(num_qubits):
                if d ==0 and n ==0 :
                    pass
                else :
                    gate_list.append([n])
                    num_gates = num_gates+1
                gate_list.append([n,(n+1)%num_qubits])
                num_gates = num_gates+1

        ##### final layer #####
        for n in range(num_qubits):
            gate_list.append([n])
            num_gates = num_gates+1

    else :
        gate_list = LayerCreator(num_qubits=num_qubits, gate_list=gate_list, double=double)
        for qubit in range(num_layer):
            gate_list = EntanglerDesigner(num_qubits=num_qubits, entangler_type=entangler_type, gate_list=gate_list)
            gate_list = LayerCreator(num_qubits=num_qubits, gate_list=gate_list, double=double)

    print("Circuit_list:", entangler_type)
    print("-->", gate_list)

    return gate_list

def EntanglerDesigner(num_qubits      : int = None,
                      entangler_type  : Optional[str]  = "cyclic",
                      gate_list       : List[int] =[]
                      ) :
    
    if entangler_type.lower() == "cyclic" :
        for n in range (num_qubits-1):
           gate_list.append([n,n+1])
        if num_qubits > 2 : 
           gate_list.append([num_qubits-1, 0])

    elif entangler_type.lower() == "ladder" :
        for n in range (0, num_qubits-1, 2):
           gate_list.append([n,n+1])
        for n in range (1, num_qubits-1, 2):
           gate_list.append([n,n+1])

    elif entangler_type.lower() == "ladder+" :
        for n in range (0, num_qubits-1, 2):
           gate_list.append([n,n+1])
        for n in range (1, num_qubits-1, 2):
           gate_list.append([n,n+1])
        gate_list.append([0,int(num_qubits/2)])


    return gate_list


def LayerCreator(num_qubits      : int = None,
                 gate_list       : List[int] =[],
                 double          : bool = False,
                 ) :

    for n in range (0, num_qubits):
       gate_list.append([n])
       if double is True :
           gate_list.append([n])

    return gate_list

def ZYcircuit(num_qubits      : int = None,
              k               : int = 0,
              ) :
    gate_list=[]
    parameter=[]
    index = 0
    for target in range(num_qubits):
        if target != k :
            gate_list.append([target])      # (Y+Z)gate
            gate_list.append([target, k])   # CNOT
            gate_list.append([k])           # Zgate
            gate_list.append([target, k])   # CNOT
            gate_list.append([target])      # (Y+Z)gate

            parameter.append([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)])
            parameter.append([0, 1, 0, 0])
            parameter.append([1, 0, 0, 1])
            parameter.append([0, 1, 0, 0])
            parameter.append([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)])

    print("gate_list:")
    print("-->", gate_list)
    print("parameter:")
    print("-->", parameter)


    return gate_list, parameter


