from __future__ import annotations

import numpy as np

from ase import units
from ase import Atoms
from ase.md.md import MolecularDynamics
from typing import Optional, List
from tqdm import tqdm
# Coefficients for the fourth-order Suzuki-Yoshida integration scheme
# Ref: H. Yoshida, Phys. Lett. A 150, 5-7, 262-268 (1990).
#      https://doi.org/10.1016/0375-9601(90)90092-3
FOURTH_ORDER_COEFFS = [
    1 / (2 - 2 ** (1 / 3)),
    -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)),
    1 / (2 - 2 ** (1 / 3)),
] 

from ase.md.npt import NPT

w = np.zeros(7)

w[0] = 0.784513610477560
w[6] = 0.784513610477560
w[1] = 0.235573213359357
w[5] = 0.235573213359357
w[2] = -1.17767998417887
w[4] = -1.17767998417887
w[3] = 1 - w[0] - w[1] - w[2] - w[4] - w[5] - w[6]


class MTTK(MolecularDynamics):
    """Isothermal molecular dynamics with Nose-Hoover chain.

    This implementation is based on the Nose-Hoover chain equations and
    the Liouville-operator derived integrator for non-Hamiltonian systems [1-3].

    - [1] G. J. Martyna, M. L. Klein, and M. E. Tuckerman, J. Chem. Phys. 97,
          2635 (1992). https://doi.org/10.1063/1.463940
    - [2] M. E. Tuckerman, J. Alejandre, R. López-Rendón, A. L. Jochim,
          and G. J. Martyna, J. Phys. A: Math. Gen. 39, 5629 (2006).
          https://doi.org/10.1088/0305-4470/39/19/S18
    - [3] M. E. Tuckerman, Statistical Mechanics: Theory and Molecular
          Simulation, Oxford University Press (2010).

    While the algorithm and notation for the thermostat are largely adapted
    from Ref. [4], the core equations are detailed in the implementation
    note available at
    https://github.com/lan496/lan496.github.io/blob/main/notes/nose_hoover_chain/main.pdf.

    - [4] M. E. Tuckerman, Statistical Mechanics: Theory and Molecular
          Simulation, 2nd ed. (Oxford University Press, 2009).
    """

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        run_steps: int,
        t_stop: float,
        p_stop: float,
        pfreq:Optional[int]=None,
        t_start: float = None,
        p_start = None,
        tfreq=0.025,
        pmode: str = "iso",
        tchain: int = 2,
        pchain: int = 2,
        **kwargs,
    ):
        """
        Parameters
        ----------
        atoms: ase.Atoms
            The atoms object.
        timestep: float
            The time step in ASE time units.
        temperature_K: float
            The target temperature in K.
        tdamp: float
            The characteristic time scale for the thermostat in ASE time units.
            Typically, it is set to 100 times of `timestep`.
        tchain: int
            The number of thermostat variables in the Nose-Hoover chain.
        tloop: int
            The number of sub-steps in thermostat integration.
        trajectory: str or None
            If `trajectory` is str, `Trajectory` will be instantiated.
            Set `None` for no trajectory.
        logfile: IO or str or None
            If `logfile` is str, a file with that name will be opened.
            Set `-` to output into stdout.
        loginterval: int
            Write a log line for every `loginterval` time steps.
        **kwargs : dict, optional
            Additional arguments passed to :class:~ase.md.md.MolecularDynamics
            base class.
        """
        super().__init__(
            atoms=atoms,
            timestep=timestep,
            **kwargs,
        )
        self.pstart = self.pstop = self.pfreq = self.p_target = self.pflag = [0]*6
        assert self.masses.shape == (len(self.atoms), 1)
        self.atoms = atoms
        self.dt = self.timestep = timestep
        self.t_stop = t_stop
        self.natom = len(self.atoms)
        # self.pflag = [0]*6
        self.pmode = pmode
        self.t_start = t_start
        if self.pmode == "iso":
            self.iso =True
            self.pflag[0]=self.pflag[1]=self.pflag[2]=1
        elif self.pmode == "x":
            self.iso = False
            self.pflag[0]=1
        elif self.pmode == "y":
            self.iso = False
            self.pflag[1]=1
        elif self.pmode == "z":
            self.iso = False
            self.pflag[2]=1
        else:
            self.pflag = [0]*6
        self.npt_flag = sum(self.pflag)
        self.pdim = self.pflag[0]+self.pflag[1]+self.pflag[2]
        self.md_pchain = pchain
        self.md_tchain = tchain
        # The following variables are updated during self.step()
        self._q = self.atoms.get_positions()
        self._p = self.atoms.get_momenta()
        self.dt2 = self.dt / 2
        self.dt4 = self.dt / 4
        self.dt8 = self.dt / 8
        self.md_tfreq = tfreq
        self.md_pfreq = pfreq
        if not self.md_pfreq:
            self.md_pfreq = 1.0 / 400 / self.dt
        if self.pmode:
            assert self.md_pfreq and self.md_tfreq       
        self.mass_peta = self.t_stop * units.kB / (self.md_pfreq)**2
        self.v_peta = np.zeros(self.md_pchain+1)
        self.nkt = (self.natom + 1) * self.t_stop * units.kB
        self.pfreq = self.md_pfreq
        self.mass_omega = self.nkt / self.pfreq / self.pfreq
        self.v_omega = np.zeros(6)
        self.tdof = 3 * self.natom
        self.run_steps = run_steps
        self.v_eta = np.zeros(self.md_tchain+1)
        self.peta = self.g_peta = [0]*self.md_pchain
        self.eta  = self.g_eta = [0]*self.md_tchain
        self.w = w
        self.nc_tchain = 1
        self.nc_pchain = 1
        self.p_start = p_start
        self.p_stop = p_stop


        if not p_start:
            self.p_start = self.p_stop
        if not t_start:
            self.t_start = self.t_stop
        self.velocities = self.atoms.get_velocities()
        self.t_target = self.get_target_temp()
        self.mass_eta = self.tdof * self.t_target * units.kB / self.md_tfreq / self.md_tfreq
        for m in range(self.md_pchain):
            self.g_peta[m] = (self.mass_peta * self.v_peta[m - 1] * self.v_peta[m - 1] - self.t_target) / self.mass_peta
        for m in range(self.md_tchain):
            self.g_eta[m] = (self.mass_eta * self.v_eta[m - 1] * self.v_eta[m - 1] - self.t_target) / self.mass_eta
        
        assert self.md_tchain>0

    def run(self, steps):
        """Perform a number of time steps."""
        for _ in range(steps):
            self.step()
            self.nsteps += 1
            self.call_observers()

    def step(self):
        self.first_half()
        self._center_of_velocities()
        self.second_half()
        self._center_of_velocities
        # print(f"dhugo:{self.dhugo}")
        # print(f"phydro:{self.p_hydro/units.GPa}")

    def _center_of_velocities(self):
        # 假设每个氢原子的速度，这里随机给出
        # atoms.set_velocities(np.array([[0.5, 0, 0], [-0.5, 0, 0]]))

        # 计算每个原子的质量
        masses = self.atoms.get_masses()


        self.velocities = self.atoms.get_velocities()

        # 计算质量加权速度
        weighted_velocities = masses[:, np.newaxis] * self.velocities

        # 计算总质量
        total_mass = np.sum(masses)
 
        # 计算质心速度
        center_of_mass_velocity = np.sum(weighted_velocities, axis=0) / total_mass

        # print("质心速度:", center_of_mass_velocity)
        self.velocities = self.velocities - center_of_mass_velocity
        self.atoms.set_velocities(self.velocities)
        

    def first_half(self) -> None:
        npt_flag = self.npt_flag
        if npt_flag:
            self.baro_thermo()
        self.t_target = self.get_target_temp()
        self.particle_thermo()
        self.t_current = self.atoms.get_temperature()
        if (npt_flag):
            # self.compute_stress()
            self.couple_stress()
            self.get_target_stress()
            self.update_baro()
            self.vel_baro()
        self.update_vel()
        if (npt_flag):
            self.update_volume()
        self.update_pos()
        if (npt_flag):
            self.update_volume()

    def second_half(self) -> None:
        npt_flag = self.npt_flag
        self.update_vel()
        if npt_flag:
            self.vel_baro()
        self.t_current = self.atoms.get_temperature()
        if npt_flag:
            # self.comute_stress()
            self.couple_stress()
            self.update_baro()
        self.particle_thermo()
        if npt_flag:
            self.baro_thermo()
    
    def get_target_stress(self):
        delta = self.nsteps/ self.run_steps
        self.p_hydro = 0
        for i in range(3):
            if self.pflag[i]:
                self.p_target[i] = self.p_start + delta * (self.p_stop - self.p_start)
                self.p_hydro += self.p_target[i]
        if self.pdim:
            self.p_hydro /= self.pdim
        # print(f"p_target:{[ptarget/units.GPa for ptarget in self.p_target]},p_hydro:{self.p_hydro/units.GPa}")

    # def setup(self):
    #     pass

    def baro_thermo(self):
        pdof = self.pdof = self.npt_flag
        self.ke_omega = 0
        self.ke_omega += sum([self.mass_omega * self.v_omega[i] * self.v_omega[i] for i in range(6) if self.pflag[i]])
        lkt_press = self.lkt_press = self.t_target * units.kB
        if not self.iso:
            self.lkt_press *= pdof  #自由度成以温度，要不要乘玻尔兹曼常数
        # print(f"g_peta:{self.g_peta},v_peta:{self.v_peta}")
        self.g_peta[0] = (self.ke_omega - lkt_press) / self.mass_peta
        scale = 1
        
        for j in range(len(self.w)):
            delta = self.w[j] * self.timestep / self.nc_pchain
            for m in range(self.md_pchain-1,0,-1):
                # print(f"m:{m},vpetashape:{len(self.v_peta)}")
                factor = np.exp(-self.v_peta[m+1] * delta / 8)
                # print(f"factor:{factor}, v_peta:{self.v_peta[m]},g_peta:{self.g_peta[m]}")
                self.v_peta[m] *= factor
                self.v_peta[m] += self.g_peta[m] * delta / 4
                self.v_peta[m] *= factor
            
            for m in range(self.md_pchain):
                self.peta[m] += self.v_peta[m] * delta / 2
            
            scale *= np.exp(-self.v_peta[0] * delta / 2)
            kecurrent = self.ke_omega*np.power(scale,2)

            self.g_peta[0] = (kecurrent - lkt_press) / self.mass_peta

            self.v_peta[0] *= factor
            self.v_peta[0] += self.g_peta[0] * delta / 4
            self.v_peta[0] *= factor

            for m in range(1,self.md_pchain):
                factor = np.exp(-self.v_peta[m+1] * delta / 8)
                self.v_peta[m] *= factor
                self.g_peta[m] = (self.mass_peta*np.power(self.v_peta[m-1],2)-self.t_target * units.kB)/self.mass_peta
                self.v_peta[m] += self.g_peta[m] * delta / 4
                self.v_peta[m] *= factor
        # print(f"v_peta:{self.v_peta}")
        self.v_omega *= scale

    def get_target_temp(self):
        delta = self.nsteps/ self.run_steps
        t_target = self.t_start + (self.t_stop - self.t_start) * delta
        # print(f"t_target:{t_target}")
        return t_target

    def particle_thermo(self):
        self.kinetic = self.atoms.get_kinetic_energy()
        # print(f"tdof:{self.tdof},t_target:{self.t_target},md_tfreq:{self.md_tfreq},md_tchain:{self.md_tchain}")
        self.mass_eta = self.tdof * self.t_target * units.kB / self.md_tfreq / self.md_tfreq
        scale = 1
        if self.mass_eta>0:
            self.g_eta[0] = (2*self.kinetic - self.tdof * self.t_target * units.kB)/self.mass_eta
        else:
            self.g_eta[0] = 0
        for i in range(len(self.w)):
            delta = self.w[i] * self.timestep / self.nc_tchain
            for m in range(self.md_tchain-1,0,-1):
                factor = np.exp(-self.v_eta[m+1] * delta / 8)
                # print(f"self.v_eta:{self.v_eta},factor:{factor},delta:{delta}")
                self.v_eta[m] *= factor
                self.v_eta[m] += self.g_eta[m] * delta / 4
                self.v_eta[m] *= factor
            
            for m in range(self.md_tchain):
                self.eta[m] += self.v_eta[m] * delta / 2

            scale *= np.exp(-self.v_eta[0] * delta / 2)
            ####检查scale是否有限
            Ke = self.kinetic *np.power(scale,2)
            if self.mass_eta>0:
                self.g_eta[0] = (2*Ke - self.tdof * self.t_target * units.kB)/self.mass_eta
            else:
                self.g_eta[0] = 0

            self.v_eta[0] *= factor
            self.v_eta[0] += self.g_eta[0] * delta / 4
            self.v_eta[0] *= factor

            for m in range(1,self.md_tchain):
                factor = np.exp(-self.v_eta[m+1] * delta / 8)
                self.g_eta[m] = (self.mass_eta*np.power(self.v_eta[m-1],2)-self.t_target * units.kB)/self.mass_eta
                self.v_eta[m] *= factor
                self.v_eta[m] += self.g_eta[m] * delta / 4
                self.v_eta[m] *= factor
        self.velocities = self.atoms.get_velocities()
        self.velocities *= scale
        self.atoms.set_velocities(self.velocities)

    def couple_stress(self):
        stress = -self.atoms.get_stress(voigt=False,include_ideal_gas=True)
        self.p_current = [0,0,0]
        if self.pmode == "iso":
            ave = np.trace(stress) / 3
            self.p_current[0] = self.p_current[1] = self.p_current[2] = ave
            print(f"p_ave{ave}")
        elif self.pmode == "x":
            self.p_current[0] = stress[0,0]
        elif self.pmode == "y":
            self.p_current[1] = stress[1,1]
        elif self.pmode == "z":
            self.p_current[2] = stress[2,2]
        else:
            raise ValueError("pmode must be iso, x, y or z")
        print(f"ave:{ave/units.GPa},p_current:{[p/units.GPa for p in self.p_current]}")
    def get_t_vector(self):
        vel = self.atoms.get_velocities()
        masses = self.atoms.get_masses()
        t_vector = np.diag([0.,0.,0.])
        for i in range(len(self.atoms)):
                t_vector += masses[i]*np.outer(vel[i],vel[i]) #不用除2吗
        return t_vector
        

    def update_baro(self):
        term_one = 0
        self.t_vector = self.get_t_vector()
        if self.iso:
            term_one = self.tdof * self.t_target * units.kB
        else:
            term_one = sum([self.t_vector[i] for i in range(len(self.t_vector)) if self.pflag[i]])
        
        term_one /= self.pdim * self.natom

        term_two = 0
        self.omega = self.atoms.get_volume()
        for i in range(3):
            if self.pflag[i]:
                self.g_omega = (self.p_current[i]- self.p_hydro) * self.omega /self.mass_omega + term_one/self.mass_omega
                self.v_omega[i] += self.g_omega * self.dt / 2
                term_two += self.v_omega[i]
        term_two /= self.pdim * self.natom
        self.mtk_term  = term_two

    def vel_baro(self):
        factor = np.diag([np.exp(-(self.v_omega[i]+self.mtk_term)*self.dt/4) for i in range(3)])
        self.velocities = self.atoms.get_velocities()
        self.velocities = np.dot(self.velocities,factor)
        if not self.iso:
            pass
        self.velocities = np.dot(self.velocities,factor)
        self.atoms.set_velocities(self.velocities)

    def update_vel(self):
        forces = self.atoms.get_forces()
        self.velocities = self.atoms.get_velocities()
        self.velocities += 0.5 * forces * self.dt / self.masses
        self.atoms.set_velocities(self.velocities)

    def update_volume(self):
        factor = np.diag([np.exp(self.v_omega[i] * self.dt / 2) for i in range(3)])
        self.cell = self.atoms.get_cell(complete=True)
        # print(f"cell:{self.cell},factor:{factor},v_omega:{self.v_omega}")
        self.cell *= factor
        self.atoms.set_cell(self.cell,scale_atoms=True)

    def update_pos(self):
        self.positions = self.atoms.get_positions()
        self.positions += self.atoms.get_velocities() * self.dt
        self.atoms.set_positions(self.positions)



class NPHugo(MTTK):
    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        # run_steps: int,
        # t_stop: float,
        p_stop: float,
        e0: float = None,
        p0: float = None,
        v0: float = None,
        pfreq:Optional[int]=None,
        tfreq=0.025,
        pmode: str = "iso",
        tchain: int = 2,
        pchain: int = 2,
        **kwargs,
    ):
        """Nose-Hoover chain thermostat for NPT ensemble.

        References:
            [1] Tuckerman, M. E., J. Alejandre, R. L. J. Grand, M. A. M. Marques, and
        """
        self.atoms = atoms
        self.pmode = pmode
        self.p0 = p0
        self.v0 = v0
        self.e0 = e0
        if not self.p0:
            self.p0 = self.atoms.get_stress(voigt=False).trace()/3
        if not self.v0:
            self.v0 = self.atoms.get_volume()
        if not self.e0:
            self.e0 = self.atoms.get_total_energy()
        self.natom = len(self.atoms)
        self.tdof = 3*self.natom
        self.t_stop = self.get_target_temp()
        self.tchain = tchain
        self.pchain = pchain
        super().__init__(atoms, timestep, run_steps=2000,
                          t_stop=self.t_stop, p_stop=p_stop, pmode=pmode,pfreq=pfreq,
                          tfreq=tfreq,tchain=tchain,pchain=pchain, **kwargs)
        self.print_init_parameters()


    def print_init_parameters(self):
        print(f"t0:{self.atoms.get_temperature()},p0:{self.p0},v0:{self.v0},e0:{self.e0},t_stop:{self.t_stop}"
              f"p_stop:{self.p_stop},p_mode:{self.pmode},tchain:{self.tchain},pchain:{self.pchain}")

    def get_target_temp(self):
        t_target = self.atoms.get_temperature() + self.compute_hugoniot()
        return t_target
        
    @property
    def volume(self):
        return self.atoms.get_volume()

    def compute_hugoniot(self):
        self.couple_stress()
        if self.pmode == "iso":
            self.p = -self.atoms.get_stress(voigt=False).trace()/3
        elif self.pmode == "x":
            self.p = -self.atoms.get_stress(voigt=False)[0, 0]
        elif self.pmode == "y":
            self.p = -self.atoms.get_stress(voigt=False)[1, 1]
        elif self.pmode == "z":
            self.p = -self.atoms.get_stress(voigt=False)[2, 2]
        dhugo = 0.5*(self.p0 + self.p)*(self.v0-self.volume)+self.e0-self.atoms.get_total_energy()
        dhugo /= self.tdof*units.kB
        self.dhugo = dhugo
        return dhugo
    
    def step(self):
        self.first_half()
        self._center_of_velocities()
        self.second_half()
        self._center_of_velocities
        print(f"dhugo:{self.dhugo}")