#!/Users/rraymondhp/miniconda3/envs/qiskit_keio/bin/python
# coding: utf-8
import numpy as np
import time
import math
import itertools
import copy
import csv
import warnings
import argparse
import sys


# import common packages
import pickle
import matplotlib as mpl
import qiskit

from qiskit import Aer
from qiskit import IBMQ
# lib from Qiskit Aqua

# lib from Qiskit tools
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.circuit.library import TwoLocal
from qiskit.providers.aer import (QasmSimulator, StatevectorSimulator, UnitarySimulator, noise)
from qiskit.quantum_info import state_fidelity, process_fidelity
from qiskit.quantum_info.operators import Operator
from qiskit.utils import QuantumInstance
from qiskit.opflow import I, X, Y, Z
from qiskit.tools.visualization import plot_histogram, plot_state_city
from lib import SequentialOptimizer, RealDeviceSelector, QubitOpLibrary, EntanglerDesigner, CircuitDesigner, PlasticPQC

import argparse
import sys

warnings.simplefilter('ignore', DeprecationWarning)

def get_backends(hub='ibm-q-internal', group='deployed', project='default'):
    """
        For internal ibm/trl hub='ibm-q-internal', group='deployed', project='default'
    """
    IBMQ.load_account()
    backends = dict()
    my_provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    pub_provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    for each in my_provider.backends(simulator=False, operational=True):
        backends[each.name()] = my_provider.get_backend(each.name())
    for each in pub_provider.backends(simulator=False, operational=True):
        backends[each.name()] = pub_provider.get_backend(each.name())
    return backends


def get_my_args(description="run experiments on real devices"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--iters", help="number of iterations", type=int, default=5)
    parser.add_argument("-j", "--jobs", help="number of independent optimizations", type=int, default=1)
    parser.add_argument("-l", "--layer",     help="layer of variational circuits", type=int, default=1)
    parser.add_argument("-b", "--backend", help="name of backend to run", type=str, default="aer_simulator_statevector")
    parser.add_argument("-m", "--method", help="name of method [fqs|fraxis|rotosolve|rotoselect]", default="fqs", type=str)
    parser.add_argument("-e", "--entangler", help="type of entangler [cyclic|ladder|ladder+|cascading]", default="cyclic", type=str)
    parser.add_argument("-g", "--group", help="group of ibm q device", default="deployed", type=str)
    parser.add_argument("-p", "--project", help="project of ibm q device", default="default", type=str)
    parser.add_argument("-s", "--shots", help="number of shots", type=int, default=8192)
    parser.add_argument("-u", "--hub", help="hub of ibm q device", default="ibm-q-internal", type=str)
    parser.add_argument("-o", "--output", help="output filename", default=None)
    parser.add_argument("-t", "--thres", help="threshold for VQE convergence", type=float, default=1e-8)
    parser.add_argument(      "--list",      help="list available backends", action='store_true')
    parser.add_argument(      "--double", help="use a gate RyRz (boolean) (default OFF)", action='store_true')
    parser.add_argument(      "--mitig", help="Error mitigation", default=True, type=bool)
    parser.add_argument(      "--method2", help="second optimization ", default=None, type=str)
    parser.add_argument(      "--pickle", help="Hamiltonian import from file", type=str, default=None)
    return parser.parse_args()



if __name__ == "__main__":
    args = get_my_args()
    if args.list is True:
        backends = get_backends(args.hub, args.group, args.project)
        print("Available backends")
        for b in backends:
            print("\t",b)
        sys.exit(0)

    if args.backend == "qasm_simulator" or "statevector_simulator" or "aer_simulator_statevector":
        backend = Aer.get_backend(args.backend) 
        quantum_instance = QuantumInstance(backend=backend, shots=args.shots)
    else:
        backends = get_backends(args.hub, args.group, args.project)
        if args.backend not in backends:
            print("The backend", args.backend, "is not available")
            sys.exit(1)
        
        backend, config, propeties = RealDeviceSelector(required_qubits=4, hub=args.hub, group=args.group, project=args.project, device=args.backend)
        if args.error is True:
            quantum_instance = QuantumInstance(backend=backend, shots=args.shots, measurement_error_mitigation_cls=CompleteMeasFitter, cals_matrix_refresh_period = 30)
        else:
            quantum_instance = QuantumInstance(backend=backend, shots=args.shots)

    ##### BENCH MARK INPUT####
    if args.pickle is None:
        qubit_op, num_qubits = QubitOpLibrary(name="maxcut-qrac", display=False)
    else :
        with open('n16_hamiltonian.pkl', 'rb') as hamiltonian:
            qubit_op = pickle.load(hamiltonian)
            num_qubits = qubit_op.num_qubits
    gate_list = CircuitDesigner(num_qubits=num_qubits, num_layer=args.layer, entangler_type=args.entangler, double=args.double)
    qr = QuantumRegister(num_qubits)
    #### END BENCHMARK INPUT ####

    # entangler_type: cyclic, ladder, (all-to-all)
    angle= None
    cparam= None

    if args.output is None:
        output_name = "maxcut_"+args.method+".xvg"
    else:
        output_name = args.output

    with open(output_name, 'a') as fp:
        for niter in range(args.jobs) :
            print("*************************")
            print("*** ", niter,"-th job ***")
            print("*************************")
            pqc = PlasticPQC(num_qubits       = num_qubits,
                             qubit_op         = qubit_op,
                             gate_list        = gate_list,
                             quantum_instance = quantum_instance,
                             cost_type        = "Expectation",
                             )
            
            pqc.set_1gate_param(method=args.method, initial_type='random')
            pqc.set_2gate_param(control_type='z')
            print("******** params   ", pqc._params)
            opt = SequentialOptimizer(pqc=pqc, threshold=args.thres, output_name=output_name)

            for _ in range(args.iters) :
                opt.search_minimum(method=args.method, max_iteration=1) 
                pqc.eval_ExactEigenSolver()
                if args.method2 is not None :
                    opt.search_minimum(method=args.method2, max_iteration=1) 
#            opt.search_minimum(method=args.method, max_iteration=1) 
            qstate = pqc.u3toStatevector(params=pqc._params)
            print(" qstate   ", qstate)
            pqc.eval_ExactEigenSolver()
            print('&', file=fp, flush=True)

