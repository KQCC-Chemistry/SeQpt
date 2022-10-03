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

""" QMolecule """

from typing import List
import os
import logging
import tempfile
import warnings
import numpy
import math
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

logger = logging.getLogger(__name__)

from qiskit.chemistry.qmolecule import QMolecule

class QMoleculeHCW(QMolecule):
    """
    Molecule data class containing driver result.

    When one of the chemistry :mod:`~qiskit.chemistry.drivers` is run and instance
    of this class is returned. This contains various properties that are made available in
    a consistent manner across the various drivers.

    Note that values here, for the same input molecule to each driver, may be vary across
    the drivers underlying code implementation. Also some drivers may not provide certain fields
    such as dipole integrals in the case of :class:`~qiskit.chemistry.drivers.PyQuanteDriver`.

    This class provides methods to save it and load it again from an HDF5 file
    """

    QMOLECULE_VERSION = 2

    def __init__(self, qmolecule):
        self.origin_driver_name    = qmolecule.origin_driver_name
        self.origin_driver_version = qmolecule.origin_driver_version 
        self.origin_driver_config  = qmolecule.origin_driver_config 

        # Energies and orbits
        self.hf_energy                = qmolecule.hf_energy                
        self.nuclear_repulsion_energy = qmolecule.nuclear_repulsion_energy 
        self.num_orbitals             = qmolecule.num_orbitals 
        self.num_alpha                = qmolecule.num_alpha 
        self.num_beta                 = qmolecule.num_beta 
        self.mo_coeff                 = qmolecule.mo_coeff 
        self.mo_coeff_b               = qmolecule.mo_coeff_b 
        self.orbital_energies         = qmolecule.orbital_energies 
        self.orbital_energies_b       = qmolecule.orbital_energies_b 

        # Molecule geometry. xyz coords are in Bohr
        self.molecular_charge = qmolecule.molecular_charge 
        self.multiplicity     = qmolecule.multiplicity 
        self.num_atoms        = qmolecule.num_atoms 
        self.atom_symbol      = qmolecule.atom_symbol 
        self.atom_xyz         = qmolecule.atom_xyz 

        # 1 and 2 electron ints in AO basis
        self.hcore   =   qmolecule.hcore   
        self.hcore_b =   qmolecule.hcore_b 
        self.kinetic =   qmolecule.kinetic 
        self.overlap =   qmolecule.overlap 
        self.eri     =   qmolecule.eri    

        # 1 and 2 electron integrals in MO basis
        self.mo_onee_ints   = qmolecule.mo_onee_ints  
        self.mo_onee_ints_b = qmolecule.mo_onee_ints_b
        self.mo_eri_ints    = qmolecule.mo_eri_ints   
        self.mo_eri_ints_bb = qmolecule.mo_eri_ints_bb
        self.mo_eri_ints_ba = qmolecule.mo_eri_ints_ba

        # Dipole moment integrals in AO basis
        self.x_dip_ints = qmolecule.x_dip_ints 
        self.y_dip_ints = qmolecule.y_dip_ints 
        self.z_dip_ints = qmolecule.z_dip_ints 

        # Dipole moment integrals in MO basis
        self.x_dip_mo_ints         = qmolecule.x_dip_mo_ints         
        self.x_dip_mo_ints_b       = qmolecule.x_dip_mo_ints_b       
        self.y_dip_mo_ints         = qmolecule.y_dip_mo_ints         
        self.y_dip_mo_ints_b       = qmolecule.y_dip_mo_ints_b       
        self.z_dip_mo_ints         = qmolecule.z_dip_mo_ints         
        self.z_dip_mo_ints_b       = qmolecule.z_dip_mo_ints_b       
        self.nuclear_dipole_moment = qmolecule.nuclear_dipole_moment  
        self.reverse_dipole_sign   = qmolecule.reverse_dipole_sign    

    @property
    def one_body_integrals(self):
        """ Returns one body electron integrals. """
        return self._one_body_integrals
#        return QMolecule.onee_to_spin(self.mo_onee_ints, self.mo_onee_ints_b)

    @one_body_integrals.setter
    def one_body_integrals(self, obi):
        """ Returns one body electron integrals. """
        self._one_body_integrals = obi
        return

    @property
    def two_body_integrals(self):
        """ Returns two body electron integrals. """
        return self._two_body_integrals
#        return QMolecule.twoe_to_spin(self.mo_eri_ints, self.mo_eri_ints_bb, self.mo_eri_ints_ba)

    @two_body_integrals.setter
    def two_body_integrals(self, tbi):
        """ Returns two body electron integrals. """
        self._two_body_integrals = tbi
        return

    @property
    def num_op(self):
        """ Returns number operator. """
        return self._num_op

    @num_op.setter
    def num_op(self, new_num_op):
        """ Returns . """
        self._num_op = new_num_op
        return

    def create_num_op(self):
        nspin_orbs = self.num_orbitals*2
        elements = [1]*nspin_orbs
        num_op = numpy.diag(elements)

        moh2_qubit = numpy.zeros([nspin_orbs, nspin_orbs, nspin_orbs, nspin_orbs])
        return num_op, moh2_qubit

    def create_partial_num_op(self, loworb, highorb):
        nspin_orbs = self.num_orbitals*2
        elements = [0]*nspin_orbs
        for i in range(loworb, highorb):
           elements[i]   = 1
           elements[i+self.num_orbitals] = 1
        num_op = numpy.diag(elements)

        moh2_qubit = numpy.zeros([nspin_orbs, nspin_orbs, nspin_orbs, nspin_orbs])
        return num_op, moh2_qubit

    @staticmethod
    def onee_to_spin(mohij, mohij_b=None, threshold=1E-12):
        """Convert one-body MO integrals to spin orbital basis

        Takes one body integrals in molecular orbital basis and returns
        integrals in spin orbitals ready for use as coefficients to
        one body terms 2nd quantized Hamiltonian.

        Args:
            mohij (numpy.ndarray): One body orbitals in molecular basis (Alpha)
            mohij_b (numpy.ndarray): One body orbitals in molecular basis (Beta)
            threshold (float): Threshold value for assignments
        Returns:
            numpy.ndarray: One body integrals in spin orbitals
        """
        if mohij_b is None:
            mohij_b = mohij

        # The number of spin orbitals is twice the number of orbitals
        norbs = mohij.shape[0]
        nspin_orbs = 2*norbs

        # One electron terms
        moh1_qubit = numpy.zeros([nspin_orbs, nspin_orbs])
        for p in range(nspin_orbs):  # pylint: disable=invalid-name
            for q in range(nspin_orbs):
                spinp = int(p/norbs)
                spinq = int(q/norbs)
                if spinp % 2 != spinq % 2:
                    continue
                ints = mohij if spinp == 0 else mohij_b
                orbp = int(p % norbs)
                orbq = int(q % norbs)
                if abs(ints[orbp, orbq]) > threshold:
                    moh1_qubit[p, q] = ints[orbp, orbq]

        return moh1_qubit

    @staticmethod
    def twoe_to_spin(mohijkl, mohijkl_bb=None, mohijkl_ba=None, threshold=1E-12):
        """Convert two-body MO integrals to spin orbital basis

        Takes two body integrals in molecular orbital basis and returns
        integrals in spin orbitals ready for use as coefficients to
        two body terms in 2nd quantized Hamiltonian.

        Args:
            mohijkl (numpy.ndarray): Two body orbitals in molecular basis (AlphaAlpha)
            mohijkl_bb (numpy.ndarray): Two body orbitals in molecular basis (BetaBeta)
            mohijkl_ba (numpy.ndarray): Two body orbitals in molecular basis (BetaAlpha)
            threshold (float): Threshold value for assignments
        Returns:
            numpy.ndarray: Two body integrals in spin orbitals
        """
        ints_aa = numpy.einsum('ijkl->ljik', mohijkl)

        if mohijkl_bb is None or mohijkl_ba is None:
            ints_bb = ints_ba = ints_ab = ints_aa
        else:
            ints_bb = numpy.einsum('ijkl->ljik', mohijkl_bb)
            ints_ba = numpy.einsum('ijkl->ljik', mohijkl_ba)
            ints_ab = numpy.einsum('ijkl->ljik', mohijkl_ba.transpose())

        # The number of spin orbitals is twice the number of orbitals
        norbs = mohijkl.shape[0]
        nspin_orbs = 2*norbs

        # The spin orbitals are mapped in the following way:
        #       Orbital zero, spin up mapped to qubit 0
        #       Orbital one,  spin up mapped to qubit 1
        #       Orbital two,  spin up mapped to qubit 2
        #            .
        #            .
        #       Orbital zero, spin down mapped to qubit norbs
        #       Orbital one,  spin down mapped to qubit norbs+1
        #            .
        #            .
        #            .

        # Two electron terms
        moh2_qubit = numpy.zeros([nspin_orbs, nspin_orbs, nspin_orbs, nspin_orbs])
        for p in range(nspin_orbs):  # pylint: disable=invalid-name
            for q in range(nspin_orbs):
                for r in range(nspin_orbs):
                    for s in range(nspin_orbs):  # pylint: disable=invalid-name
                        spinp = int(p/norbs)
                        spinq = int(q/norbs)
                        spinr = int(r/norbs)
                        spins = int(s/norbs)
                        if spinp != spins:
                            continue
                        if spinq != spinr:
                            continue
                        if spinp == 0:
                            ints = ints_aa if spinq == 0 else ints_ba
                        else:
                            ints = ints_ab if spinq == 0 else ints_bb
                        orbp = int(p % norbs)
                        orbq = int(q % norbs)
                        orbr = int(r % norbs)
                        orbs = int(s % norbs)
                        if abs(ints[orbp, orbq, orbr, orbs]) > threshold:
                            moh2_qubit[p, q, r, s] = -0.5*ints[orbp, orbq, orbr, orbs]

        return moh2_qubit

    def create_one_body_integrals(self):
        return QMolecule.onee_to_spin(self.mo_onee_ints, self.mo_onee_ints_b)

    def modify_one_body_integrals(self, maxorb, threshold=1E-12, factor=0.0):
        """ Returns one body electron integrals. """
        mohij = self.mo_onee_ints
        mohij_b = self.mo_onee_ints_b
        if mohij_b is None:
            mohij_b = mohij

        # The number of spin orbitals is twice the number of orbitals
        norbs = mohij.shape[0]
        nspin_orbs = 2*norbs

        # One electron terms
        moh1_qubit = numpy.zeros([nspin_orbs, nspin_orbs])
        count=0
        n = max(maxorb) 

        for p in range(nspin_orbs):  # pylint: disable=invalid-name
            for q in range(nspin_orbs):
                spinp = int(p/norbs)
                spinq = int(q/norbs)
                if spinp % 2 != spinq % 2:
                    continue
                ints = mohij if spinp == 0 else mohij_b
                orbp = int(p % norbs)
                orbq = int(q % norbs)
                flag = False
                if( orbp <= n and orbq <= n ):
                    for m in maxorb : 
                        if (orbp-m)*(orbq-m)== 0 :
                            flag = True
                    if flag and abs(ints[orbp,orbq]) >0.0 :
                        count+=1

        dcount=math.ceil(count*factor)
        top = numpy.zeros(dcount, dtype=float) 
        print('one integral: factor           % 6f' % factor)
        print('one integral: non-zero terms   % 6d' % count)
        print('one integral: counted terms    % 6d' % dcount)

        for p in range(nspin_orbs):  # pylint: disable=invalid-name
            for q in range(nspin_orbs):
                spinp = int(p/norbs)
                spinq = int(q/norbs)
                if spinp % 2 != spinq % 2:
                    continue
                ints = mohij if spinp == 0 else mohij_b
                orbp = int(p % norbs)
                orbq = int(q % norbs)
                if( orbp <= n and orbq <= n ):
                    flag = False
                    for m in maxorb : 
                        if (orbp-m)*(orbq-m)== 0 :
                            flag = True
                    if flag== True and ints[orbp, orbq] > top[dcount-1]:
                        top[dcount-1] = ints[orbp,orbq]
                        top = sorted(top, key=float, reverse=True)

        threshold = top[dcount-1]
        threshold = 0.0
                          
        print('one integral: threshold        % 6e' % threshold)

        for p in range(nspin_orbs):  # pylint: disable=invalid-name
            for q in range(nspin_orbs):
                spinp = int(p/norbs)
                spinq = int(q/norbs)
                if spinp % 2 != spinq % 2:
                    continue
                ints = mohij if spinp == 0 else mohij_b
                orbp = int(p % norbs)
                orbq = int(q % norbs)
                if( orbp <= n and orbq <= n ):
                    flag = False
                    for m in maxorb : 
                        if (orbp-m)*(orbq-m)== 0 :
                            flag = True
                    if flag :
                        moh1_qubit[p, q] = ints[orbp, orbq]*factor
#                        print('wt orbp={} orbq={} m={}  {}'.format(orbp,orbq,m,moh1_qubit[p,q]))
                    else :
                        moh1_qubit[p, q] = ints[orbp, orbq]
#                        print('wo orbp={} orbq={} m={}  {}'.format(orbp,orbq,m, moh1_qubit[p,q]))
#        print('moh1_qubit={}'.format(moh1_qubit))
        return moh1_qubit


    def create_two_body_integrals(self):
        return QMolecule.twoe_to_spin(self.mo_eri_ints, self.mo_eri_ints_bb, self.mo_eri_ints_ba)

    def modify_two_body_integrals(self, maxorb, threshold=0.0, factor=0.0):
        """Convert two-body MO integrals to spin orbital basis

        Takes two body integrals in molecular orbital basis and returns
        integrals in spin orbitals ready for use as coefficients to
        two body terms in 2nd quantized Hamiltonian.

        Args:
            mohijkl (numpy.ndarray): Two body orbitals in molecular basis (AlphaAlpha)
            mohijkl_bb (numpy.ndarray): Two body orbitals in molecular basis (BetaBeta)
            mohijkl_ba (numpy.ndarray): Two body orbitals in molecular basis (BetaAlpha)
            threshold (float): Threshold value for assignments
        Returns:
            numpy.ndarray: Two body integrals in spin orbitals
        """
        mohijkl = self.mo_eri_ints
        mohijkl_bb = self.mo_eri_ints_bb
        mohijkl_ba = self.mo_eri_ints_ba

        ints_aa = numpy.einsum('ijkl->ljik', mohijkl)

        if mohijkl_bb is None or mohijkl_ba is None:
            ints_bb = ints_ba = ints_ab = ints_aa
        else:
            ints_bb = numpy.einsum('ijkl->ljik', mohijkl_bb)
            ints_ba = numpy.einsum('ijkl->ljik', mohijkl_ba)
            ints_ab = numpy.einsum('ijkl->ljik', mohijkl_ba.transpose())

        # The number of spin orbitals is twice the number of orbitals
        norbs = mohijkl.shape[0]
        nspin_orbs = 2*norbs

        # The spin orbitals are mapped in the following way:
        #       Orbital zero, spin up mapped to qubit 0
        #       Orbital one,  spin up mapped to qubit 1
        #       Orbital two,  spin up mapped to qubit 2
        #            .
        #            .
        #       Orbital zero, spin down mapped to qubit norbs
        #       Orbital one,  spin down mapped to qubit norbs+1
        #            .
        #            .
        #            .

        # Two electron terms
        n = max(maxorb) 
        moh2_qubit = numpy.zeros([nspin_orbs, nspin_orbs, nspin_orbs, nspin_orbs])
        count=0
        for p in range(nspin_orbs):  # pylint: disable=invalid-name
            for q in range(nspin_orbs):
                for r in range(nspin_orbs):
                    for s in range(nspin_orbs):  # pylint: disable=invalid-name
                        spinp = int(p/norbs)
                        spinq = int(q/norbs)
                        spinr = int(r/norbs)
                        spins = int(s/norbs)
                        if spinp != spins:
                            continue
                        if spinq != spinr:
                            continue
                        if spinp == 0:
                            ints = ints_aa if spinq == 0 else ints_ba
                        else:
                            ints = ints_ab if spinq == 0 else ints_bb
                        orbp = int(p % norbs)
                        orbq = int(q % norbs)
                        orbr = int(r % norbs)
                        orbs = int(s % norbs)
                        if( orbp <= n and orbq <= n and  orbr <= n and orbs <= n ):
                            flag = False
                            for m in maxorb : 
                                if (orbp-m)*(orbq-m)*(orbr-m)*(orbs-m) == 0 :
                                    flag = True
                            if flag :
                                if ints[orbp, orbq, orbr, orbs] > 0.0 :
                                    count+=1

        dcount=math.ceil(count*factor)
        top = numpy.zeros(dcount, dtype=float) 

        for p in range(nspin_orbs):  # pylint: disable=invalid-name
            for q in range(nspin_orbs):
                for r in range(nspin_orbs):
                    for s in range(nspin_orbs):  # pylint: disable=invalid-name
                        spinp = int(p/norbs)
                        spinq = int(q/norbs)
                        spinr = int(r/norbs)
                        spins = int(s/norbs)
                        if spinp != spins:
                            continue
                        if spinq != spinr:
                            continue
                        if spinp == 0:
                            ints = ints_aa if spinq == 0 else ints_ba
                        else:
                            ints = ints_ab if spinq == 0 else ints_bb
                        orbp = int(p % norbs)
                        orbq = int(q % norbs)
                        orbr = int(r % norbs)
                        orbs = int(s % norbs)
                        if( orbp <= n and orbq <= n and  orbr <= n and orbs <= n ):
                            flag = False
                            for m in maxorb : 
                                if (orbp-m)*(orbq-m)*(orbr-m)*(orbs-m) == 0 :
                                    flag = True
                            if flag :
                                    top[dcount-1] = ints[orbp, orbq, orbr, orbs]
                                    top = sorted(top, key=float, reverse=True)

        threshold = top[dcount-1]

        print('two integral: factor           % 6f' % factor)
        print('two integral: non-zero terms   % 6d' % count)
        print('two integral: counted terms    % 6d' % dcount)
        print('two integral: threshold        % 6e' % threshold)

        for p in range(nspin_orbs):  # pylint: disable=invalid-name
            for q in range(nspin_orbs):
                for r in range(nspin_orbs):
                    for s in range(nspin_orbs):  # pylint: disable=invalid-name
                        spinp = int(p/norbs)
                        spinq = int(q/norbs)
                        spinr = int(r/norbs)
                        spins = int(s/norbs)
                        if spinp != spins:
                            continue
                        if spinq != spinr:
                            continue
                        if spinp == 0:
                            ints = ints_aa if spinq == 0 else ints_ba
                        else:
                            ints = ints_ab if spinq == 0 else ints_bb
                        orbp = int(p % norbs)
                        orbq = int(q % norbs)
                        orbr = int(r % norbs)
                        orbs = int(s % norbs)
                        if( orbp <= n and orbq <= n and  orbr <= n and orbs <= n ):
                            flag = False
                            for m in maxorb : 
                                if (orbp-m)*(orbq-m)*(orbr-m)*(orbs-m) == 0 :
                                    flag = True
                            if flag :
                                    moh2_qubit[p, q, r, s] = -0.5*ints[orbp, orbq, orbr, orbs]*factor
                            else :
                                moh2_qubit[p, q, r, s] = -0.5*ints[orbp, orbq, orbr, orbs]
#                                if abs(ints[orbp, orbq, orbr, orbs]) > threshold:
#                                    moh2_qubit[p, q, r, s] = -0.5*ints[orbp, orbq, orbr, orbs]
#                                        moh2_qubit[p, q, r, s] = -0.5*ints[orbp, orbq, orbr, orbs]*factor

        return moh2_qubit


