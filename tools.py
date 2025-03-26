from ase import Atoms
from ase.io import read,write
from typing import List
import numpy as np
from mattersim.forcefield import MatterSimCalculator
from ase.optimize import LBFGS
def select_by_force(traj:List[Atoms],threshold:float=50):
    """select the frames with max force larger than threshold"""
    forces = [traj[i].get_forces() for i in range(len(traj))]
    maxforces = [np.max(forces[i]) for i in range(len(forces))]
    index = [i for i in range(len(maxforces)) if maxforces[i] < threshold]
    traj = [traj[i] for i in index]
    return traj


def small_opt(file:str):
    atoms = read(file)
    calculator=MatterSimCalculator(device="cuda")
    atoms._calc = calculator
    opt = LBFGS(atoms)
    opt.run(fmax=0.05,steps=200)
    write(f"{file.split('.')[0]}_opt.pdb",atoms)