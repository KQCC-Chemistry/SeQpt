# SeQpt Variational Quantum Algorithm with Sequantial Quantum Optimizer## 
How to run the program for Heisenberg model. 

The main programs is prepared for the respective target systems such as "heisenberg.py", "maxcut.py" , although it is not nessarily. 
It means it is possible that you can preapre a single main program generalized for any system.
In the following, we show an example to execute sequential quantum optimizer with heisenberg.py

You can list the options to run the program as below:
```
(py375-qk036) $ python heisenberg.py -h
usage: heisenberg.py [-h] [-q QUBITS] [-l LAYER] [-j JOBS] [-i ITERS]
                     [-o OUTPUT] [--initial_type INITIAL_TYPE]
                     [--initial_axis INITIAL_AXIS] [--double] [-b BACKEND]
                     [-s SHOTS] [-m METHOD] [-g GROUP] [-p PROJECT] [-u HUB]
                     [-e ENTANGLER] [-t THRES] [--method2 METHOD2]
                     [--initial INITIAL] [--list] [--error ERROR]

optional arguments:
  -h, --help                    show this help message and exit
  -q QUBITS, --qubits QUBITS    number of qubits
  -l LAYER, --layer LAYER       layer of variational circuits
  -j JOBS, --jobs JOBS          number of independent optimizations
  -i ITERS, --iters ITERS       number of iterations
  -o OUTPUT, --output OUTPUT    output filename
  --initial_type INITIAL_TYPE   Generator for initial condition [state-random|parameter-random|perturb]
  --initial_axis INITIAL_AXIS   Generator for initial condition [x|y|z]
  --double                      use a gate RyRz (boolean) (default OFF)
  -b BACKEND, --backend BACKEND name of backend to run
  -s SHOTS, --shots SHOTS       number of shots
  -m METHOD, --method METHOD    name of method [control-fqs|fqs|fraxis|rotosolve|rotoselect]
  -g GROUP, --group GROUP       group of ibm q device
  -p PROJECT, --project PROJECT project of ibm q device
  -u HUB, --hub HUB             hub of ibm q device
  -e ENTANGLER, --entangler ENTANGLER
                                name of entangler
  -t THRES, --thres THRES       threshold for VQE convergence
  --method2 METHOD2             name of method [control-fqs|fqs|fraxis|rotosolve|rotoselect]
  --initial INITIAL             initial parameter [random|perturb]
  --list                        list available backends
  --error ERROR                 mitigation error

```

## Execute Rotosolve/Fraxis/FQS 
This is a repository for running sequential quantum optimizer for PQC. Here we show an example to execute sequential quantum optimizer with heisenberg.py
```
$ python heisenberg.py -q 5 -m fqs -i 10 -j 1 --entangler cyclic
```

Recipe for FQS input.
## (1) provide Hamiltonian
A class qubit_op_library may be helpful to make a gate_list.
```
qubit_op = QubitOpLibrary(num_qubits=args.qubits, name=heisenberg, entangler_type=args.entangler)
```

## (2) provide gate_list
Example: gate_list= [0,1,2,[0,1],[1,2], ....]
where the elements stand for the qubit index. If the paired index with [..] represents 2-qubit gates.
A class CircuitDesigner may be helpful to make a gate_list.
```
gate_list = CircuitDesigner(num_qubits=args.qubits, num_layer=args.layer, entangler_type=args.entangler)
```

## (3) make a PlasticPQC
```
gate_list = PlasticPQC(num_qubits=args.qubits, qubit_op=qubit_op, entangler_type=args.entangler)
```

## (4) make a PlasticPQC
```
pqc = PlasticPQC(num_qubits=args.qubits, qubit_op=qubit_op, entangler_type=args.entangler)
```

## (5) do optimization
```
opt = SequentialOptimizer(pqc=pqc, output_name=output_name)
final_val = opt.search_minimum(method=args.method, target_list=target_list, max_iteration=args.iters)
```
