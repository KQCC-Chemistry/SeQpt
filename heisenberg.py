#!/Users/rraymondhp/miniconda3/envs/qiskit_keio/bin/python
# coding: utf-8
import numpy as np
import time
import math
import itertools
import copy
import csv


# import common packages
import pickle
import matplotlib as mpl
import qiskit

from qiskit import Aer
from qiskit import IBMQ
# lib from Qiskit Aqua

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.circuit.library import TwoLocal
from qiskit.providers.aer import (QasmSimulator, StatevectorSimulator, UnitarySimulator, noise)
from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit.quantum_info import state_fidelity, process_fidelity
from qiskit.quantum_info.operators import Operator
from qiskit.utils import QuantumInstance
from qiskit.tools.visualization import plot_histogram, plot_state_city
#from qiskit.opflow import I, X, Y, Z

# lib from Qiskit tools
from lib import PlasticPQC
from lib import RealDeviceSelector, QubitOpLibrary, EntanglerDesigner, CircuitDesigner, ZYcircuit, SequentialOptimizer



import argparse
import sys

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
    parser.add_argument("-q", "--qubits",    help="number of qubits", type=int, default=5)
    parser.add_argument("-l", "--layer",     help="layer of variational circuits", type=int, default=1)
    parser.add_argument("-j", "--jobs",      help="number of independent optimizations", type=int, default=20)
    parser.add_argument("-i", "--iters",     help="number of iterations", type=int, default=100)
    parser.add_argument("-o", "--output",    help="output filename", default=None)
    # Specific to Fraxis,Roto
    parser.add_argument(      "--initial_type", help="Generator for initial condition [state-random|parameter-random|perturb]", type=str, default="state-random")
    # Specific to Roto
    parser.add_argument(      "--initial_axis", help="Generator for initial condition [x|y|z]", type=str, default="y")
    parser.add_argument(      "--double", help="use a gate RyRz (boolean) (default OFF)", action='store_true')
    # Specific to device/simulator
    parser.add_argument("-b", "--backend",   help="name of backend to run", type=str, default="statevector_simulator")
    parser.add_argument("-s", "--shots",     help="number of shots", type=int, default=8192)
    parser.add_argument("-m", "--method",    help="name of method [control-fqs|fqs|fraxis|rotosolve|rotoselect]", default="fqs", type=str)
    parser.add_argument("-g", "--group",     help="group of ibm q device", default="deployed", type=str)
    parser.add_argument("-p", "--project",   help="project of ibm q device", default="default", type=str)
    parser.add_argument("-u", "--hub",       help="hub of ibm q device", default="ibm-q-internal", type=str)
    parser.add_argument("-e", "--entangler", help="name of entangler", type=str, default="cyclic")
    parser.add_argument("-t", "--thres",     help="threshold for VQE convergence", type=float, default=1e-8)
    parser.add_argument(      "--method2",   help="name of method [control-fqs|fqs|fraxis|rotosolve|rotoselect]", default=None, type=str)
    parser.add_argument(      "--initial",   help="initial parameter [random|perturb]", type=str, default="random")
    parser.add_argument(      "--list",      help="list available backends", action='store_true')
    parser.add_argument(      "--error",     help="mitigation error", default=True, type=bool)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_my_args()
    if args.list is True:
        backends = get_backends(args.hub, args.group, args.project)
        print("Available backends")
        for b in backends:
            print("\t",b)
        sys.exit(0)

    if args.backend == "statevector_simulator" or "qasm_simulator":
        backend = Aer.get_backend(args.backend) 
        quantum_instance = QuantumInstance(backend=backend, shots=args.shots)
    else:
        backends = get_backends(args.hub, args.group, args.project)
        if args.backend not in backends:
            print("The backend", args.backend, "is not available")
            sys.exit(1)
        
        backend, config, propeties = RealDeviceSelector(required_qubits=args.qubits, hub=args.hub, group=args.group, project=args.project, device=args.backend)
        if args.error is True:
            quantum_instance = QuantumInstance(backend=backend, shots=args.shots, measurement_error_mitigation_cls=CompleteMeasFitter, cals_matrix_refresh_period = 30)
        else:
            quantum_instance = QuantumInstance(backend=backend, shots=args.shots)

    ##### BENCH MARK INPUT####
    print(" QUBITS   ", args.qubits)
    qubit_op, num_qubits = QubitOpLibrary(num_qubits=args.qubits, name="heisenberg", coeff=[1.0, 1.0], display=True)

    #### END BENCHMARK INPUT ####

#    gate_list = []
#    target_list = []
    barrier_list = []

    gate_list = CircuitDesigner(num_qubits=num_qubits, num_layer=args.layer, entangler_type=args.entangler, double=args.double)
    print(gate_list)


    angle   = None
    cparam  = None
    u3param = None
    cu3param = None

    if args.output is None:
        output_name = "heisenberg_"+args.method.lower()+"_"+str(args.layer)+".xvg"
    else:
        output_name = args.output


    print("initial_type-",args.initial)
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
            
            print("******** params   ", pqc._params)
            if args.method.lower() == "control-fqs" :
                target_list = pqc._list_2gate
            else :
                target_list = pqc._list_1gate
            
            if args.double is True :
                n=0
                for target in target_list :
                    if n%2 == 0 :
                        pqc._params[target] = pqc.gen_param_roto(initial_type=args.initial, axis='y')
                    else :
                        pqc._params[target] = pqc.gen_param_roto(initial_type=args.initial, axis='z')
                    n=n+1
            else:
                pqc.set_1gate_param(method=args.method, initial_type=args.initial)

            pqc.set_2gate_param(control_type='z')
            print("******** params ******\n", pqc._params)
            opt = SequentialOptimizer(pqc=pqc, threshold=args.thres, output_name=output_name)
            
            for _ in range(args.iters) :
                opt.search_minimum(method=args.method, max_iteration=1) 
                pqc.eval_ExactEigenSolver()
                if args.method2 is not None :
                    opt.search_minimum(method=args.method2, max_iteration=1) 

            final_val = opt.search_minimum(method=args.method, target_list=target_list, max_iteration=args.iters) 

            qstate = pqc.u3toStatevector(params=pqc._params)
            print(" qstate   ", qstate)
            pqc.eval_ExactEigenSolver()
            print('&', file=fp, flush=True)



