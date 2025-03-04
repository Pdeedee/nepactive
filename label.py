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

# 确保使用 'spawn' 启动方法来避免 CUDA 初始化问题
def change_calc(atoms_list: List[Atoms], device, pot_file, traj_file):
    # 设置当前进程的CUDA设备
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device)  # 使用设备编号
    # torch.cuda.set_device(device)
    
    calculator = MatterSimCalculator(load_path=pot_file, device=f"cuda:{device}")
    traj = Trajectory(traj_file, mode='a')

    for atoms in tqdm(atoms_list):
        atoms._calc = calculator
        atoms.get_potential_energy()
        traj.write(atoms)

# 在 Linux 或其他平台上，也需要加上 if __name__ == "__main__": 保护
if __name__ == "__main__":
    # 加载原子数据
    atoms_list = read("test.xyz", index=":")
    pot_file = "best_model.pth"
    train_ratio = 1
    tqdm_use = True

    # 设置多进程启动方式
    multiprocessing.set_start_method('spawn', force=True)

    # 将 atoms_list 划分为 4 部分
    num_parts = 4
    split_atoms = [atoms_list[i::num_parts] for i in range(num_parts)]

    with ProcessPoolExecutor(max_workers=num_parts) as executor:
        futures = []
        for i, part in enumerate(split_atoms):
            traj_file = f"candidate_{i}.traj"  # 为每个子进程创建不同的 traj 文件
            futures.append(executor.submit(change_calc, part, i, pot_file, traj_file))  # 提交任务并分配显卡

        # 等待所有任务完成
        for future in futures:
            future.result()  # 可以根据需要处理异常

    # 读取Trajectory对象中的原子信息并分类
    atoms = read("candidate.traj", index=":")
    train: List[Atoms] = []
    test: List[Atoms] = []
