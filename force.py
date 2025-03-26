from concurrent.futures import ProcessPoolExecutor
import torch
from mattersim.forcefield import MatterSimCalculator
from ase.io import read, write
from ase import Atoms
from typing import List, Optional
from ase.io import Trajectory
import random
from tqdm import tqdm
import multiprocessing
import os
from pynep.calculate import NEP

# 确保使用 'spawn' 启动方法来避免 CUDA 初始化问题
def get_force_list(atoms_list: List[Atoms], nep_file):

    f_list = []
    for atoms in tqdm(atoms_list):
        atoms._calc = NEP(model_file = nep_file)
        forces = atoms.get_forces()
        energy = atoms.get_potential_energy()
        f_list.append((forces, energy))
        # traj.write(atoms)
    return f_list

def force_main(atoms_list:List[Atoms],pot_files:list):
    # 设置多进程启动方式
    multiprocessing.set_start_method('spawn', force=True)
    with ProcessPoolExecutor(max_workers=len(pot_files)) as executor:
        futures:List[ProcessPoolExecutor] = []
        futures = [executor.submit(get_force_list, atoms_list, pot_file) for pot_file in pot_files]
        # f_lists = [future.result() for future in futures]
    # return f_lists
        f_lists = []
        for future in futures:
            try:
                result = future.result()
                f_lists.append(result)
            except Exception as e:
                print(f"Error processing future: {e}")
    return f_lists



# 在 Linux 或其他平台上，也需要加上 if __name__ == "__main__": 保护
if __name__ == "__main__":
    force_main()
