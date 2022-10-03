#!/usr/bin/env python
from qiskit.providers.ibmq import least_busy
from qiskit import IBMQ
import sys  

def RealDeviceSelector(required_qubits : int = 2,
                       hub     = 'ibm-q-utokyo',
                       group   = 'keio-internal',
                       project = 'keio-main',
                       device = None
                       ) :
    
    # %% Get information of Real Device
    IBMQ.load_account()
    print(IBMQ.providers())
    provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    backends = provider.backends(operational=True, local=False, simulator=False, filters=lambda x: x.configuration().n_qubits >= required_qubits)
    
    # %%
    if device is None :
       for backend in backends:
           print(backend)
           conf = backend.configuration()
           print("n qubits", conf.n_qubits)
           print("qv", conf.quantum_volume)
       # %%
       maxqv = max([(x.configuration().quantum_volume) for x in backends])
       backends = [x for x in backends if x.configuration().quantum_volume == maxqv]
       # %%
       backend = least_busy(backends)

    else :
       backend = provider.get_backend(device)

    conf = backend.configuration()
    prop = backend.properties()
    print("Selected Backend=", backend)

    return backend, conf, prop
