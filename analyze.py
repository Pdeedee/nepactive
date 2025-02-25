import numpy as np
from ase import io
from ase.neighborlist import NeighborList
from collections import defaultdict
from typing import List, Dict
from collections import Counter
from ase.io.vasp import read_vasp
from ase.neighborlist import NeighborList, natural_cutoffs
import os

def identify_molecules_in_frame(atoms) -> List[str]:
    """
    识别每一帧中的分子，并根据原子种类及数量生成分子名称。
    """
    visited = set()  # 用于记录已访问的原子索引
    molecules = []   # 用于存储识别到的分子

    # 基于共价半径为每个原子生成径向截止
    cutoffs = natural_cutoffs(atoms, mult=0.8)
    
    # 获取成键原子，考虑周期性边界条件
    nl = NeighborList(cutoffs=cutoffs, bothways=True, self_interaction=False)
    nl.update(atoms)  # 更新邻居列表
    
    # 遍历所有原子
    for i in range(len(atoms)):
        if i not in visited:  # 如果当前原子尚未被访问
            current_molecule = defaultdict(int)  # 用于统计元素及其数量
            stack = [i]  # 使用栈进行深度优先搜索，初始化栈为当前原子索引
            while stack:
                atom_index = stack.pop()  # 从栈中取出一个原子索引
                if atom_index not in visited:
                    visited.add(atom_index)  # 标记为已访问
                    atom_symbol = atoms[atom_index].symbol  # 获取原子的元素符号
                    current_molecule[atom_symbol] += 1  # 统计该元素的数量
                    # 获取与当前原子成键的原子索引
                    bonded_indices, _ = nl.get_neighbors(atom_index)
                    stack.extend(idx for idx in bonded_indices if idx not in visited)

            # 将当前分子信息添加到分子列表中
            if current_molecule:
                molecules.append(current_molecule)

    # 将分子的元素信息转换为分子名称（C7N8O2 等形式）
    molecule_names = []
    for molecule in molecules:
        molecule_name = ''.join(f"{element}{molecule[element]}" for element in sorted(molecule.keys()))
        molecule_names.append(molecule_name)

    return molecule_names

def analyze_trajectory(trajectory_file: str):
    """
    分析ASE轨迹文件中的每一帧，识别每一帧中的分子及其原子数量。
    """
    # 读取轨迹文件
    if trajectory_file.endswith('.xyz'):
        traj = io.read(trajectory_file, format="extxyz", index=':')
    else:
        traj = io.read(trajectory_file, index=':')

    # 遍历轨迹的每一帧
    for frame_idx, atoms in enumerate(traj):
        print(f"Analyzing frame {frame_idx + 1}...")
        
        # 识别该帧中的分子
        molecule_names = identify_molecules_in_frame(atoms)

        # 统计每种分子出现的次数
        molecule_counts = Counter(molecule_names)
        
        # 输出该帧中的分子信息
        print(f"Frame {frame_idx + 1} contains {len(molecule_counts)} distinct molecules:")
        for molecule, count in molecule_counts.items():
            print(f"  {molecule}: {count} molecule(s)")

