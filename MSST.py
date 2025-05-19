### get_cell 是否与想象中一致，是否为转置
###
import weakref

import numpy as np
from ase import Atoms,units
from ase.md.md import MolecularDynamics
#from ase.md import MDLogger,npt
# from ase import units
# import sys
import time

class MSST(MolecularDynamics):
    def __init__(self, atoms: Atoms,timestep: float,  loginterval, vbox:np.ndarray, shock_direction:str='y',
                   v0 = 0, p0 = 0,trajectory=None, logfile=None, qmass:int = 7e7,tscale = 0.01, 
                     v_shock = 8e3,append_trajectory:bool = False):
        MolecularDynamics.__init__(self, atoms, timestep, trajectory,
                                   logfile, loginterval, append_trajectory = append_trajectory)
        #general parameters
        loginterval: int = 1
        self.interval:int = 10
        self.atoms = atoms
        self.dt = timestep
        if shock_direction == 'x':
            self.sd = 0
        elif shock_direction == 'y':
            self.sd = 1
        elif shock_direction == 'z':
            self.sd = 2
        #MSST parameters
        self.initialized = 0
        self.v_shock =  v_shock * units.m / units.second 
        self.V0 = v0# Initial volume
        self.total_mass = atoms.get_masses().sum()  #mass convert
        self.atomnumbers = atoms.get_global_number_of_atoms()
        self.qmass = qmass 
        # self.bias = bias    # Scaling factor for energy conservation adjustments
        self.h0 = self.atoms.get_cell().T############################h与h0和晶体矩阵为转置关系
        self.velocity = self.atoms.get_velocities() 
        self.positions = self.atoms.get_positions()  #!!!!!!!!!!!!!!!!pos输出的是不是A!!!!!!!!!!!!!!!!!!!!!
        self.alpha = np.outer(self.vbox[:,0],np.linalg.inv(self.vbox)[0,:])
        self.beta = np.outer(self.vbox[:,1],np.linalg.inv(self.vbox)[1,:])+np.outer(self.vbox[:,2],np.linalg.inv(self.vbox)[2,:])
        self.gamma:float=1
        self.gammadot:float = 0.0
       
        self.hdot=self.h0@self.alpha*self.gammadot
        self.P0=p0

        self.h = self.h0
        self.V = np.abs(np.linalg.det(self.h))
        self.q = np.dot(np.linalg.inv(self.h),self.atoms.get_positions().T).T##############列向量
         # a = 0.5*self.qmass*self.gammadot*self.gammadot*self.V*self.V + atoms.get_total_energy()
        if tscale is not None:
            a=0.5*self.qmass*self.gammadot*self.gammadot*self.V*self.V/self.total_mass
            b= atoms.get_total_energy()
            hamiltonian = a+b
            print(f"hamiltonian:{hamiltonian},a:{a},b:{b}")
            # fac1 = tscale * self.total_mass * 2.0 / (atoms.get_kinetic_energy() * self.qmass)
            fac1 = tscale * atoms.get_kinetic_energy() *2.0*self.total_mass / (self.qmass * self.V*self.V)
            self.gammadot:float = -1.0 * np.sqrt(fac1)
            atoms.set_velocities(atoms.get_velocities()*np.sqrt((1.0-tscale)))
            a=0.5*self.qmass*self.gammadot*self.gammadot*self.V*self.V/self.total_mass
            b= atoms.get_total_energy()
            hamiltonian = a+b
            print(f"hamiltonian:{hamiltonian},a:{a},b:{b}")
            # print(f"kineticenergy:{atoms.get_kinetic_energy()},fac1:{fac1}, gammadot:{self.gammadot},qmass:{self.qmass},mass:{self.total_mass}")
        else:
            self.gammadot:float = 0.0
        self.zero_center_of_mass_momentum()

    @property
    def Hamiltonian(self):
        ori_energy = self.atoms.get_total_energy()
        box_pot = -0.5*self.total_mass*np.power(self.v_shock,2)*np.power((1-self.gamma),2)-self.vbox[1,:].T@self.P0@self.vbox[1,:]*(self.V0-self.volume)/(self.vbox[1,:].T@self.vbox[1,:])
        box_kin = 0.5*self.qmass*self.gammadot*self.gammadot*self.volume*self.volume/(2*self.total_mass) 
        Hamiltonian = box_kin+ori_energy+box_pot
        H = [Hamiltonian,ori_energy,box_pot,box_kin]
        return H

    @property
    def volume(self):
        return self.atoms.get_volume()

    @property
    def cell_length_a(self):
        # return self.msst_direction@self.atoms.get_cell()@self.msst_direction
        return self.gamma

    def run(self,steps):
        # if not self.initialized:
        #     self.initialize()
        # print(f"cell_length_a:{self.cell_length_a:.7f},alpha:{self.alpha},beta:{self.beta},\n gamma: {self.gamma:.4f}, gammadot: {self.gammadot:.9f}, V: {self.V:.4f}")
        self.iter=0
        for i in range(steps):
            # start_time = time.time()
            self.step()
            self.zero_center_of_mass_momentum()
            self.nsteps += 1
            print(f"gamma:{self.cell_length_a},gammadot:{self.gammadot},hamiltonian:{self.Hamiltonian}") 
            self.call_observers()

    def attach(self, function, interval=10, *args, **kwargs):
        """Attach callback function or trajectory.

        At every *interval* steps, call *function* with arguments
        *args* and keyword arguments *kwargs*.

        
        If *function* is a trajectory object, its write() method is
        attached, but if *function* is a BundleTrajectory (or another
        trajectory supporting set_extra_data(), said method is first
        used to instruct the trajectory to also save internal
        data from the NPT dynamics object.
        """
        if hasattr(function, "set_extra_data"):
            # We are attaching a BundleTrajectory or similar
            function.set_extra_data("npt_init",
                                    WeakMethodWrapper(self, "get_init_data"),
                                    once=True)
            function.set_extra_data("npt_dynamics",
                                    WeakMethodWrapper(self, "get_data"))
        MolecularDynamics.attach(self, function, interval, *args, **kwargs)
           
    def step(self):
        self.first_half()
        self.second_half()
            
    def first_half(self):
        self.dthalf = self.dt/2

        self.propagate_voldot()
        self.vsum = np.square(self.atoms.get_velocities()).sum(axis=1).sum()
        oldv = self.atoms.get_velocities()
        self.propagate_vel()
        self.vsum = np.square(self.atoms.get_velocities()).sum(axis=1).sum()
        self.atoms.set_velocities(oldv)
        self.propagate_vel()
        self.update_half_volume()
        self.update_pos()
        self.update_half_volume()

    def second_half(self):
        self.propagate_vel()
        self.vsum = np.square(self.atoms.get_velocities()).sum(axis=1).sum()
        self.propagate_voldot()

    
    def get_center_of_mass_momentum(self):
        "Get the center of mass momentum."
        return self.atoms.get_momenta().sum(0)

    def _getnatoms(self):
        """Get the number of atoms.

        In a parallel simulation, this is the total number of atoms on all
        processors.
        """
        return len(self.atoms)
    
    def zero_center_of_mass_momentum(self, verbose=0):
        "Set the center of mass momentum to zero."
        cm = self.get_center_of_mass_momentum()
        abscm = np.sqrt(np.sum(cm * cm))
        if verbose and abscm > 1e-4:
            self._warning(
                self.classname +
                ": Setting the center-of-mass momentum to zero "
                "(was %.6g %.6g %.6g)" % tuple(cm))
        self.atoms.set_momenta(self.atoms.get_momenta() -
                               cm / self._getnatoms())
        self.qdot = np.dot(np.linalg.inv(self.h),self.atoms.get_velocities().T).T
        self.q = np.dot(np.linalg.inv(self.h),self.atoms.get_positions().T).T

    def update_half_volume(self):
        self.gamma += self.dthalf*self.gammadot
        self.h=self.h0 @ (self.alpha*self.gamma+self.beta)
        self.atoms.set_cell(self.h.T,scale_atoms=True)

    def update_q(self):#msst的文献里，受力与体积变化的速度有关，为何在位置更新的步骤中没体现出来
        self.q+=self.dt*self.qdot
        r = np.dot(self.q, self.h.T)
        self.atoms.set_positions(r)
        
    def propagate_gammadot(self):
        alpha = self.alpha
        beta =self.beta
        gamma=self.gamma
        self.h = self.h0 @ (alpha*gamma+beta)#!!!注意是否全局修改
        h0 = self.h0
        h = self.h
        h_inv = np.linalg.inv(h)
        dthalf = self.dt / 2
        totmass = self.total_mass
        V = np.abs(np.linalg.det(h))

        Q=self.qmass
        V0 = self.V0
        p0 = self.P0
        v_shock = self.v_shock  # 原始 m/s
        stress = self.atoms.get_stress(voigt=False,include_ideal_gas=True)

        I = np.eye(3)
        X = h0 @ alpha * V @ h_inv
        f = np.trace(X)
        A = totmass/(Q*np.power(f,2))*\
            np.trace((-stress-p0*I-np.power(v_shock,2)*totmass/V0*(1-V/V0)*I) @ X)#stress的正负要搞对

        self.gammadot = self.gammadot + dthalf*A


    def propagate_qdot(self):
        forces = self.atoms.get_forces()

        h=self.h
        h_inv=np.linalg.inv(h)
        hdot=self.h0@self.alpha*self.gammadot
        dthalf = self.dt/2
        self.qdot = self.atoms.get_velocities()@(h_inv.T)
        masses = self.atoms.get_masses()[:, np.newaxis]  # 将质量数组改为 (1344, 1)
        N = (forces / masses) @ (h_inv.T)
        '''
        用了ETD一阶算法，线性项比较大的时候用指数形式，非线性项比较小的时候用ETD1阶的二阶泰勒展开
        '''
        for i in np.arange(len(self.atoms)):
            L=-h_inv@(np.transpose(hdot@h_inv)+hdot@h_inv)@h@(self.qdot[i,:].T)
            for j in np.arange(3):
                l=L[j]/self.qdot[i,j]
                n=N[i,j]+L[(j+1)%3]+L[(j+2)%3]########################
                if np.abs(l * dthalf) > 1e-6:    
                    expd = np.exp(dthalf * l)
                    self.qdot[i,j] = expd * (n + l * self.qdot[i,j] - n/expd) / l
                else:
                    self.qdot[i,j] += (n + l * self.qdot[i,j]) * dthalf+ 0.5 * (np.power(l,2) * self.qdot[i,j] + n*l)\
                          * np.power(dthalf,2)


    def get_data(self):
        "Return data needed to restore the state."
        return {'volume': self.volume,
                'pressure':self.stress,
                'gamma': self.gamma,
                'cell_length_a': self.cell_length_a,
                }


class WeakMethodWrapper:
    """A weak reference to a method.

    Create an object storing a weak reference to an instance and
    the name of the method to call.  When called, calls the method.

    Just storing a weak reference to a bound method would not work,
    as the bound method object would go away immediately.
    """

    def __init__(self, obj, method):
        self.obj = weakref.proxy(obj)
        self.method = method

    def __call__(self, *args, **kwargs):
        m = getattr(self.obj, self.method)
        return m(*args, **kwargs)
