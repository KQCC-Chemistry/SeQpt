#!/Users/rraymondhp/miniconda3/envs/qiskit_keio/bin/python
# coding: utf-8
import numpy as np
import time
import math
import itertools
import copy
import csv
import argparse, sys


# import common packages
import pickle
import matplotlib as mpl
import qiskit

from qiskit import Aer
# lib from Qiskit Aqua

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.circuit.library import TwoLocal
from qiskit.providers.aer import (QasmSimulator, StatevectorSimulator, UnitarySimulator, noise)
from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit.quantum_info import state_fidelity, process_fidelity

#from qiskit.aqua import Operator, QuantumInstance
from qiskit.utils import QuantumInstance
from qiskit.algorithms import NumPyEigensolver, EigensolverResult

# lib from Qiskit Aqua Chemistry
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriverType, ElectronicStructureMoleculeDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber, ElectronicEnergy
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer
from qiskit.opflow import I, X, Y, Z

# lib from Qiskit tools
from qiskit.tools.visualization import plot_histogram, plot_state_city
from lib import SequentialOptimizer, RealDeviceSelector, QubitOpLibrary, EntanglerDesigner, CircuitDesigner, PlasticPQC, LayerwisePQC




def get_my_args(description="run h2sto3g experiments on real devices"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--iters", help="number of iterations", type=int, default=10)
    parser.add_argument("-t", "--thres", help="threshold for VQE convergence", type=float, default=1e-8)
    parser.add_argument("-j", "--jobs", help="number of independent optimizations", type=int, default=1)
    parser.add_argument("-b", "--backend", help="name of backend to run", type=str, default="statevector_simulator")
    parser.add_argument("-l", "--layer", help="layer of variational circuits", type=int, default=1)
    parser.add_argument("-o", "--output",    help="output filename", default=None)
#    parser.add_argument("-b", "--backend", help="name of backend to run", type=str, default="aer_simulator_statevector")
    parser.add_argument("-s", "--espot", help="Entangler spot [inner|outer]", default="outer", type=str)
    parser.add_argument("-f", "--final", help="Don`t append Final Layer", action='store_false')
    parser.add_argument("-e", "--entangler", help="name of entangler", type=str, default="ladder+")
    parser.add_argument("-m", "--method", help="name of method [control-fqs|fqs|fraxis|rotosolve|rotoselect]", default="fqs", type=str)
    # Specific to device/simulator
    parser.add_argument(      "--project", help="project of ibm q device", default="default", type=str)
    parser.add_argument(      "--list", help="list available backends", action='store_true')
    parser.add_argument(      "--group", help="group of ibm q device", default="deployed", type=str)
    parser.add_argument(      "--shots", help="number of shots", type=int, default=8192)
    parser.add_argument(      "--hub", help="hub of ibm q device", default="ibm-q-internal", type=str)
    parser.add_argument(      "--error", help="mitigation error", action='store_true')
    # Specific to Fraxis,Roto
    parser.add_argument(      "--initial_type", help="Generator for initial condition [state-random|parameter-random|perturb]", type=str, default="state-random")
    # Specific to Roto
    parser.add_argument(      "--initial_axis", help="Generator for initial condition [x|y|z]", type=str, default="y")
    parser.add_argument(      "--double", help="use a gate RyRz (boolean) (default OFF)", action='store_true')
    # Specific to Quantum Chemistry
    parser.add_argument(      "--mapping", help="mapping [parity|jordan_wigner|bravyi_kitaev]", type=str, default="parity")
    parser.add_argument(      "--basis", help="basis [sto-3g|6-31g]", type=str, default="6-31g")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_my_args()
    if args.list is True:
        backends = get_backends(args.hub, args.group, args.project)
        print("Available backends")
        for b in backends:
            print("\t",b)
        sys.exit(0)

    if args.backend == "statevector_simulator" or "qasm_simulator" or "aer_simulator_statevector":
        backend = Aer.get_backend(args.backend) 
        quantum_instance = QuantumInstance(backend=backend, shots=args.shots)
    else:
        backends = get_backends(args.hub, args.group, args.project)
        if args.backend not in backends:
            print("The backend", args.backend, "is not available")
            sys.exit(1)
        
        backend, config, propeties = RealDeviceSelector(required_qubits=2, hub=args.hub, group=args.group, project=args.project, device=args.backend)
        if args.error is True:
            quantum_instance = QuantumInstance(backend=backend, shots=args.shots, measurement_error_mitigation_cls=CompleteMeasFitter, cals_matrix_refresh_period = 30)
        else:
            quantum_instance = QuantumInstance(backend=backend, shots=args.shots)


    
    ###############################
    #### Control parameter ########
    ###############################
    ##  output file name ###

    ## sto-3g, 6-31g
    pBASIS = args.basis
    #H, He atom: 1s 
    #Li-Ne:    : 1s + 2s + 2p*3 

    #       H  L
    #orbID  0, 1, 2, 3 
    #acc    2, 0, 0, 0
    #AS     0, 1, 2, 3
    pADIABATIC=True
    pACTIVESPACE = [0,1,2,3,4,5] # 6-31g full
    pNMOLECULARORB = 4

    ## UCCSD, RY, RYRZ, SwapRZ
    pMAXITER = args.iters
    pSHOTS   = args.shots

    ## jordan_wigner, parity, bravyi_kitaev
    pMAPPING = args.mapping
    p2QREDUCTION = True

    ## statevector_simulator or qasm_simulator
    #pSIMULATOR = "statevector_simulator"
    pSIMULATOR = "qasm_simulator"

    ## statevector_simulator or qasm_simulator

    pHF= True

    ## cost function type: energy, variance, mix
    pCTYPE = "energy"

    pDEVICE    = args.backend

    ###############################
    #### END Control parameter ####
    ###############################
    if pMAPPING != "parity" :
        p2QREDUCTION = False


###
    ##  new trial with equilibrium #
    molecule = Molecule(geometry=[['H', [0.0, 0.0, 0.0]],['H', [0.0, 0.0, 0.75]]], charge=0, multiplicity=1)
    driver = ElectronicStructureMoleculeDriver(molecule, basis='6-31g', driver_type=ElectronicStructureDriverType.PYSCF)    
    properties         = driver.run()
    particle_number    = properties.get_property(ParticleNumber)
    electronic_energy  = properties.get_property(ElectronicEnergy)
    active_orbitals    = pACTIVESPACE 
    active_space_trafo = ActiveSpaceTransformer(num_electrons=particle_number.num_particles, num_molecular_orbitals=pNMOLECULARORB)
    
    es_problem    = ElectronicStructureProblem(driver, transformers=[active_space_trafo])
    second_q_op   = es_problem.second_q_ops()
    num_electrons = sum(particle_number.num_particles)
    print(second_q_op[0])

    if pMAPPING == "jordan_wigner":
        qubit_converter = QubitConverter(mapper=JordanWignerMapper())
    elif pMAPPING == "parity":
        qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=p2QREDUCTION)
    elif pMAPPING == "bravyi_kitaev":
        qubit_converter = QubitConverter(mapper=BravyiKitaevMapper())
    else:
        print("pMAPPING wrong")
        sys.exit(1)

    qubit_op = qubit_converter.convert(second_q_op[0], num_particles=es_problem.num_particles)


    HF_energy =  electronic_energy._reference_energy - electronic_energy._nuclear_repulsion_energy 
    print('   number of electrons               : %  12d' % num_electrons)
    print('   HF total energy                   : % .12f' % HF_energy)
    print('   HF electronic energy              : % .12f' % electronic_energy._reference_energy)
    print('   Nuclear_repulsion_energy          : % .12f' % electronic_energy._nuclear_repulsion_energy)


    num_qubits = second_q_op[0]._register_length 
    if pHF :
        if p2QREDUCTION :
            num_qubits -= 2
        qr = QuantumRegister(num_qubits)
        pre_circuit  = QuantumCircuit(qr)
        flag_parity = False
        for i in range(num_qubits):
            if flag_parity is True:
                if i%(num_qubits//2) < num_electrons//2 :
                    flag_parity = False
                else :
                    pre_circuit.x(qr[i])
                    flag_parity = True
            else :
                if i%(num_qubits//2) < num_electrons//2 :
                    pre_circuit.x(qr[i])
                    flag_parity = True
#                    pre_circuit.x(qr[i+ num_qubits//2])
                else :
                    flag_parity = False

    else:
        if p2QREDUCTION :
            num_qubits -= 2
        qr = QuantumRegister(num_qubits)
        pre_circuit  = QuantumCircuit(qr)
        parameters = np.zeros(num_qubits*3*(pDEPTH+1))

    print(pre_circuit)


#    gate_list = CircuitDesigner(num_qubits=num_qubits, num_layer=1, entangler_type=args.entangler)
    

    angle= None
    cparam= None
    u3param= None
    cu3param= None

    if args.output is None:
        output_name = args.method+"_d"+".xvg"
    else:
        output_name = args.output

#    gate_list=[[0,1],[1],[1,0],[3,4],[4],[4,3],[0,1],[1,2],[2],[2,1],[1,0],[3,4],[4,5],[5],[5,4],[4,3],[0,3],[3],[3,0]]
#    target_list1 = [1,4,8,13,17]
#    target_list2 = [0,2,3,5,6,7,9,10,11,12,14,15,16,17,18]

    gate_list = []
    target_list = []
    barrier_list = []
    num_gates = 0

    ##### initial layer
    for n in range(num_qubits):
        gate_list.append([n])
        target_list.append(num_gates)
        num_gates = num_gates+1

    for d in range(args.layer):
        for n in range(num_qubits):
            if d ==0 and n ==0 :
                pass
            else :
                gate_list.append([n])
                target_list.append(num_gates)
                num_gates = num_gates+1
            gate_list.append([n,(n+1)%num_qubits])
            num_gates = num_gates+1

    ##### final layer
    for n in range(num_qubits):
        gate_list.append([n])
        target_list.append(num_gates)
        num_gates = num_gates+1

    print(gate_list)


    qr = QuantumRegister(num_qubits)
    pre_circuit = QuantumCircuit(qr)
    pre_circuit.x(0)
    pre_circuit.x(3)

    with open(output_name, 'a') as fp:
        for niter in range(args.jobs) :
            pqc = PlasticPQC(num_qubits       = num_qubits,
                             qubit_op         = qubit_op,
                             gate_list        = gate_list,
                             quantum_instance = quantum_instance,
                             pre_circuit      = pre_circuit,
                             cost_type        = "Expectation",
                             )

            pqc.set_1gate_param(method=args.method, initial_type='random')
            pqc.set_2gate_param(control_type='z')
            print("******** param****   ")
            print(pqc._params)
            opt = SequentialOptimizer(pqc=pqc, threshold=args.thres, output_name=output_name)
            opt.search_minimum(method=args.method, target_list=target_list, max_iteration=10) 
            pqc.eval_ExactEigenSolver()

            pqc.eval_ExactEigenSolver()


            print('&', file=fp, flush=True)



