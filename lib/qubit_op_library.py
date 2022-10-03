#!/usr/bin/env python
import sys  
from typing import Optional, List
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow import I, X, Y, Z

#This function stores qubit operators which were used in our papers 
def Pauli_local(num_qubits: int=None, 
                 pauli_list: tuple=None
                 ):
    paulis=''
    for i in range(num_qubits):
        try: 
            paulis = paulis+pauli_list[i]
        except:
            paulis = paulis+'I'
#    return Pauli(paulis)
    return PauliOp(Pauli(paulis))


#    test =Pauli('IXYZ')
#    test2= PauliOp(Pauli('IXYZ'))
#    test3= PauliOp(Pauli('XXXX'))
#    test4=test2+test3
#    return PauliOp(Pauli('IXYZ'))


def QubitOpLibrary(name      : str = None,
                  num_qubits : int = None,              
                  coeff      : int = None,              
                  display    : Optional[bool] = False              
                  )-> OperatorBase :
    
    if name is None :
        qubit_op = None

    elif name.lower() == "maxcut-qrac" :
        num_qubits = 4
        qubit_op =-0.5*((I^I^I^I) - 3.0 * (X^I^X^I))\
                  -0.5*((I^I^I^I) - 3.0 * (X^I^Y^I))\
                  -0.5*((I^I^I^I) - 3.0 * (X^I^Z^I))\
                  -0.5*((I^I^I^I) - 3.0 * (Y^I^Z^I))\
                  -0.5*((I^I^I^I) - 3.0 * (Y^I^I^X))\
                  -0.5*((I^I^I^I) - 3.0 * (Y^I^I^Y))\
                  -0.5*((I^I^I^I) - 3.0 * (Z^I^Y^I))\
                  -0.5*((I^I^I^I) - 3.0 * (Z^I^I^X))\
                  -0.5*((I^I^I^I) - 3.0 * (Z^I^I^Z))\
                  -0.5*((I^I^I^I) - 3.0 * (I^X^X^I))\
                  -0.5*((I^I^I^I) - 3.0 * (I^X^I^Y))\
                  -0.5*((I^I^I^I) - 3.0 * (I^X^I^Z))\
                  -0.5*((I^I^I^I) - 3.0 * (I^I^X^X))\
                  -0.5*((I^I^I^I) - 3.0 * (I^I^Y^Y))\
                  -0.5*((I^I^I^I) - 3.0 * (I^I^Z^Z))

    elif name.lower() == "maxcut-raw" :
        num_qubits = 10
        qubit_op =-0.5*((I^I^I^I^I^I^I^I^I^I) - (Z^I^I^I^I^Z^I^I^I^I))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (I^Z^I^I^I^I^Z^I^I^I))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (I^I^Z^I^I^I^I^Z^I^I))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (I^I^I^Z^I^I^I^I^Z^I))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (I^I^I^I^Z^I^I^I^I^Z))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (Z^I^Z^I^I^I^I^I^I^I))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (I^I^Z^I^Z^I^I^I^I^I))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (I^Z^I^I^Z^I^I^I^I^I))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (I^Z^I^Z^I^I^I^I^I^I))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (Z^I^I^Z^I^I^I^I^I^I))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (I^I^I^I^I^Z^Z^I^I^I))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (I^I^I^I^I^I^Z^Z^I^I))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (I^I^I^I^I^I^I^Z^Z^I))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (I^I^I^I^I^I^I^I^Z^Z))\
                  -0.5*((I^I^I^I^I^I^I^I^I^I) - (I^I^I^I^I^Z^I^I^I^Z))

    elif name.lower() == "maxcut-raw16" :
        num_qubits = 16
        qubit_op = 0.664 * Z^I^I^I^I^I^I^I^I^I^I^I^I^I^I^Z \
                 + 0.982 * I^I^I^I^I^I^I^I^I^I^I^I^I^Z^I^Z \
                 + 0.444 * I^I^I^I^I^I^I^I^I^Z^I^I^I^I^I^Z \
                 + 0.189 * I^I^I^I^Z^I^I^I^I^I^I^I^I^I^I^Z \
                 + 0.191 * I^Z^I^I^I^I^I^I^I^I^I^I^I^I^I^Z \
                 + 0.901 * I^I^Z^I^I^I^I^I^I^I^I^I^I^I^Z^I \
                 + 0.589 * I^I^I^I^I^I^I^I^I^I^I^Z^I^I^Z^I \
                 - 0.402 * I^I^I^I^I^I^Z^I^I^I^I^I^I^I^Z^I \
                 - 0.857 * I^I^I^I^I^I^I^I^Z^I^I^I^I^Z^I^I \
                 - 0.888 * I^I^I^I^I^I^I^I^I^I^I^I^Z^Z^I^I \
                 - 0.455 * I^I^I^I^I^I^I^I^I^Z^I^I^I^Z^I^I \
                 + 0.39  * I^I^Z^I^I^I^I^I^I^I^I^I^I^Z^I^I \
                 + 0.174 * I^I^I^I^I^I^I^I^I^I^Z^I^Z^I^I^I \
                 + 0.291 * I^I^I^I^I^I^I^I^I^I^I^Z^Z^I^I^I \
                 + 0.942 * I^I^I^I^I^I^I^I^I^Z^I^I^Z^I^I^I \
                 - 0.004 * I^I^I^I^I^Z^I^I^I^I^I^I^Z^I^I^I \
                 + 0.99  * I^I^I^I^Z^I^I^I^I^I^I^I^Z^I^I^I \
                 - 0.54  * I^Z^I^I^I^I^I^I^I^I^I^I^Z^I^I^I \
                 - 0.595 * I^I^I^I^I^I^I^I^I^Z^I^Z^I^I^I^I \
                 + 0.171 * Z^I^I^I^I^I^I^I^I^I^I^Z^I^I^I^I \
                 - 0.26  * I^I^I^I^I^I^Z^I^I^I^Z^I^I^I^I^I \
                 + 0.705 * I^I^I^I^I^I^I^I^Z^I^Z^I^I^I^I^I \
                 - 0.867 * I^I^I^I^I^Z^I^I^I^I^Z^I^I^I^I^I \
                 - 0.251 * I^I^I^Z^I^I^I^I^I^I^Z^I^I^I^I^I \
                 - 0.492 * I^I^Z^I^I^I^I^I^I^I^Z^I^I^I^I^I \
                 - 0.122 * I^Z^I^I^I^I^I^I^I^I^Z^I^I^I^I^I \
                 + 0.211 * Z^I^I^I^I^I^I^I^I^I^Z^I^I^I^I^I \
                 + 0.707 * Z^I^I^I^I^I^I^I^I^Z^I^I^I^I^I^I \
                 - 0.999 * I^I^I^I^Z^I^I^I^I^Z^I^I^I^I^I^I \
                 + 0.078 * I^I^I^Z^I^I^I^I^I^Z^I^I^I^I^I^I \
                 - 0.875 * I^I^I^Z^I^I^I^I^Z^I^I^I^I^I^I^I \
                 - 0.908 * I^I^Z^I^I^I^I^I^Z^I^I^I^I^I^I^I \
                 + 0.066 * I^Z^I^I^I^I^I^Z^I^I^I^I^I^I^I^I \
                 + 0.433 * I^I^I^I^I^I^Z^Z^I^I^I^I^I^I^I^I \
                 + 0.329 * I^I^I^I^I^Z^I^Z^I^I^I^I^I^I^I^I \
                 + 0.555 * I^I^I^I^Z^I^I^Z^I^I^I^I^I^I^I^I \
                 - 0.821 * I^I^I^Z^I^I^I^Z^I^I^I^I^I^I^I^I \
                 - 0.895 * Z^I^I^I^I^I^I^Z^I^I^I^I^I^I^I^I \
                 + 0.466 * I^I^Z^I^I^I^Z^I^I^I^I^I^I^I^I^I \
                 + 0.081 * I^I^I^I^Z^I^Z^I^I^I^I^I^I^I^I^I \
                 + 0.563 * I^I^I^Z^I^I^Z^I^I^I^I^I^I^I^I^I \
                 + 0.824 * I^Z^I^I^I^I^Z^I^I^I^I^I^I^I^I^I \
                 + 0.039 * Z^I^I^I^I^Z^I^I^I^I^I^I^I^I^I^I \
                 - 0.219 * I^I^I^Z^I^Z^I^I^I^I^I^I^I^I^I^I \
                 - 0.009 * I^I^Z^I^I^Z^I^I^I^I^I^I^I^I^I^I \
                 - 0.348 * I^Z^I^I^I^Z^I^I^I^I^I^I^I^I^I^I \
                 + 0.703 * I^I^I^Z^Z^I^I^I^I^I^I^I^I^I^I^I \
                 - 0.843 * Z^I^I^I^Z^I^I^I^I^I^I^I^I^I^I^I \
                 + 0.27  * I^I^Z^Z^I^I^I^I^I^I^I^I^I^I^I^I \
                 + 0.013 * Z^I^Z^I^I^I^I^I^I^I^I^I^I^I^I^I \
                 - 0.555 * I^Z^Z^I^I^I^I^I^I^I^I^I^I^I^I^I \
                 + 0.766 * Z^Z^I^I^I^I^I^I^I^I^I^I^I^I^I^I

    elif name.lower() in [ "heisenberg-n", "heisebergn", "n-heisenberg", "nheisenberg"]:
        if num_qubits is None:
            raise QiskitError("you have to specify num_qubits for Heisenberg-N")
        c0=1.0
        c1=1.0
#        for operator in enumerate(X^X, Y^Y, Z^Z):
        for operator in [X^X, Y^Y, Z^Z]:
            for left in range(num_qubits-2):
                import pdb; pdb.set_trace()
                for _, in range(left):
                    import pdb; pdb.set_trace()
                    operator = I.tensor(operator)
                for _, in range(num_qubits-2-left):
                    import pdb; pdb.set_trace()
                    operator = operator.tensor(I)
            import pdb; pdb.set_trace()
            try:
                qubit_op = qubit_op + c0*operator
            except:
                qubit_op =  c0*operator

    elif name.lower() == "heisenberg" :
        qubit_op  = 0.0*PauliOp(Pauli('I'*num_qubits))
        for i  in range(num_qubits) :
            qubit_op = qubit_op + coeff[0]*Pauli_local(num_qubits, pauli_list={i%num_qubits: 'X', (i+1)%num_qubits:'X'})
            qubit_op = qubit_op + coeff[0]*Pauli_local(num_qubits, pauli_list={i%num_qubits: 'Y', (i+1)%num_qubits:'Y'})
            qubit_op = qubit_op + coeff[0]*Pauli_local(num_qubits, pauli_list={i%num_qubits: 'Z', (i+1)%num_qubits:'Z'})
        for i  in range(num_qubits) :
            qubit_op = qubit_op + coeff[1]*Pauli_local(num_qubits, pauli_list={i: 'Z'})


    elif name.lower() == "local2" :
        qubit_op = Z^Z
        for q in range(num_qubits-2) :
             qubit_op = qubit_op^I 

    #print("entangler:", entangler_type)
    #print("-->", entangler)
    if display is True :
        print(qubit_op)
        print(qubit_op.to_matrix_op())

    return qubit_op, num_qubits



