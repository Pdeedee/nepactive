import os
import time
import shutil
import logging
import subprocess
from typing import List, Dict
from collections import defaultdict
from ase.io.vasp import read_vasp
from ase.neighborlist import NeighborList, natural_cutoffs


class CONTCAR_processing:

    def __init__(self, target_folder='optimized'):
        # 获取脚本的当前目录
        self.base_dir = os.path.dirname(__file__)
        # 寻找同一目录下的optimized文件夹
        self.folder_dir = os.path.join(self.base_dir, target_folder)
        self.max_density_dir = os.path.join(self.folder_dir, 'max_density')
        self.primitive_cell_dir = os.path.join(self.folder_dir, 'primitive_cell')
        logging.info(f'Processing CONTCAR in {target_folder}')

    def _identify_molecules(self, atoms, check_N5=True, count_N5=2) -> List[Dict[str, int]]:
        visited = set()  # 用于记录已经访问过的原子索引
        molecules = []   # 用于存储识别到的独立分子
        # 基于共价半径为每个原子生成径向截止
        # threshold = 0.48
        # cutoffs = [threshold] * len(atoms)
        cutoffs = natural_cutoffs(atoms, mult=0.7)
        # 获取成键原子，考虑周期性边界条件
        nl = NeighborList(cutoffs=cutoffs, bothways=True, self_interaction=False)
        nl.update(atoms)  # 更新邻居列表
        # 遍历所有原子
        for i in range(len(atoms)):
            # 如果当前原子尚未被访问
            if i not in visited:
                current_molecule = defaultdict(int)  # 用于统计元素及其数量
                stack = [i]  # 使用栈进行深度优先搜索，初始化栈为当前原子索引
                # 深度优先搜索
                while stack:
                    atom_index = stack.pop()  # 从栈中取出一个原子索引
                    if atom_index not in visited:
                        visited.add(atom_index)  # 标记为已访问
                        atom_symbol = atoms[atom_index].symbol  # 获取原子的元素符号
                        current_molecule[atom_symbol] += 1  # 统计该元素的数量
                        # 获取与当前原子成键的原子索引
                        bonded_indices, _ = nl.get_neighbors(atom_index)
                        # 将未访问的成键原子索引添加到栈中
                        stack.extend(idx for idx in bonded_indices if idx not in visited)
                # 如果当前分子包含元素信息，则将其添加到分子列表中
                if current_molecule:
                    molecules.append(current_molecule) 
        merged_molecules = defaultdict(int)  # 用于合并分子及其计数
        for molecule in molecules:
            # 将分子信息转换为可哈希的元组形式，以便合并
            molecule_tuple = frozenset(molecule.items())
            merged_molecules[molecule_tuple] += 1  # 计数相同的分子
        # 设置标志表示 N5 环的检测结果，0表示并未进行检测，1表示有完整 N5 环，-1表示无完整 N5 环
        N5_found, N5_flag = False, 0
        if check_N5:
            # 检查是否存在 N5 分子
            for molecule, count in merged_molecules.items():
                # 确保只有氮元素且数量为 5
                if (dict(molecule).get('N', 0) == 5 and len(molecule) == 1 and count == count_N5):
                    N5_found = True
                    break
            N5_flag = 1 if N5_found else -1
        # 返回合并后的分子及其数量, flag 标志表示 N5 环的检测结果
        return merged_molecules, N5_flag

    def _molecules_information(self, molecules: List[Dict[str, int]]):
        """
        Set the output format of the molecule. Output simplified element information in the specified order of C, N, O, H, which may include other elements.
        """
        # 定义固定顺序的元素
        fixed_order = ['C', 'N', 'O', 'H']
        logging.info('Identified independent molecules:')
        for idx, (molecule, count) in enumerate(molecules.items()):
            molecule = dict(molecule)
            total_atoms = sum(molecule.values())  # 计算当前分子的原子总数
            # 构建输出字符串
            output = []
            for element in fixed_order:
                if element in molecule:
                    output.append(f"{element} {molecule[element]}") 
            # 如果有其他元素，添加到输出中
            for element in molecule:
                if element not in fixed_order:
                    output.append(f"{element} {molecule[element]}")
            formatted_output =  ' '.join(output)
            logging.info(f'Molecule {idx + 1} (Total Atoms: {total_atoms}, Count: {count}): {formatted_output}')

    def _sequentially_read_files(self, dir, prefix_name='POSCAR_'):
        """
        Private method: 
        Extract numbers from file names, convert them to integers, sort them by sequence, and return a list containing both indexes and file names
        """
        # 获取dir文件夹中所有以prefix_name开头的文件，在此实例中为POSCAR_
        files = [f for f in os.listdir(dir) if f.startswith(prefix_name)]
        file_index_pairs = []
        for filename in files:
            index_part = filename[len(prefix_name):]  # 选取去除前缀'POSCAR_'的数字
            if index_part.isdigit():  # 确保剩余部分全是数字
                index = int(index_part)
                file_index_pairs.append((index, filename))
        file_index_pairs.sort(key=lambda pair: pair[0])
        return file_index_pairs
    
    def read_density_and_sort(self, n=10, N5_screen=True, count_N5=2, detail_log=False):
        """
        Obtain the atomic mass and unit cell volume from the optimized CONTCAR file, and obtain the ion crystal density. Finally, take n CONTCAR files with the highest density and save them separately for viewing.
        
        :param n: 取前n个最大密度的文件
        """
        os.chdir(self.base_dir)
        # 获取所有以'CONTCAR_'开头的文件，并按数字顺序处理
        CONTCAR_file_index_pairs = self._sequentially_read_files(self.folder_dir, prefix_name='CONTCAR_')
        # 逐个处理文件
        density_index_list = []
        for _, CONTCAR_filename in CONTCAR_file_index_pairs:
            atoms = read_vasp(os.path.join(self.folder_dir, CONTCAR_filename))
            if N5_screen:
                molecules, flag = self._identify_molecules(atoms, check_N5=True, count_N5=count_N5)
                if flag == 1:
                    if detail_log:
                        logging.info(f'{CONTCAR_filename} with independent N5 molecules')
                        self._molecules_information(molecules)
                    atoms_volume = atoms.get_volume()  # 体积单位为立方埃（Å³）
                    atoms_masses = sum(atoms.get_masses())  # 质量单位为原子质量单位(amu)
                    # 1.66054这一转换因子用于将原子质量单位转换为克，以便在宏观尺度上计算密度g/cm³
                    density = 1.66054 * atoms_masses / atoms_volume       
                    density_index_list.append((density, CONTCAR_filename))
            else:
                atoms_volume = atoms.get_volume()  # 体积单位为立方埃（Å³）
                atoms_masses = sum(atoms.get_masses())  # 质量单位为原子质量单位(amu)
                # 1.66054这一转换因子用于将原子质量单位转换为克，以便在宏观尺度上计算密度g/cm³
                density = 1.66054 * atoms_masses / atoms_volume       
                density_index_list.append((density, CONTCAR_filename))
        if N5_screen:
            print(f'Total optimized ionic crystals: {len(CONTCAR_file_index_pairs)}')
            print(f'Screened ionic crystals with N5: {len(density_index_list)}')
            logging.info(f'Total optimized ionic crystals: {len(CONTCAR_file_index_pairs)}')
            logging.info(f'Screened ionic crystals with N5: {len(density_index_list)}')
        # 根据密度降序排序
        sorted_filename = sorted(density_index_list, key=lambda x: x[0], reverse=True)
        # 将前n个最大密度的CONTCAR文件进行重命名并保存到max_density文件夹
        os.makedirs(self.max_density_dir, exist_ok=True)
        for density, CONTCAR_filename in sorted_filename[:n]:
            # 生成新的包含密度值的文件名，并重命名文件
            # 密度转换为字符串，保留4位小数
            density_str = f'{density:.4f}'
            # 保留 CONTCAR 的序数信息，方便回推检查
            number = CONTCAR_filename.split("_")[1]
            OUTCAR_filename = f'OUTCAR_{number}'
            new_CONTCAR_filename = f'CONTCAR_{density_str}_{number}'
            new_OUTCAR_filename = f'OUTCAR_{density_str}_{number}'
            shutil.copy(f'{self.folder_dir}/{CONTCAR_filename}', f'{self.max_density_dir}/{new_CONTCAR_filename}')
            shutil.copy(f'{self.folder_dir}/{OUTCAR_filename}', f'{self.max_density_dir}/{new_OUTCAR_filename}')
            print(f'New CONTCAR and OUTCAR of {density_str}_{number} are renamed and saved')
            logging.info(f'New CONTCAR and OUTCAR of {density_str}_{number} are renamed and saved')

    def phonopy_processing_max_density(self, specific_directory=None):
        """
        Use phonopy to check and generate symmetric primitive cells, reducing the complexity of subsequent optimization calculations, and preventing pyxtal.from_random from generating double proportioned supercells. 
        """
        if specific_directory:
            self.phonopy_dir = os.path.join(self.base_dir, specific_directory)
            self.primitive_cell_dir = os.path.join(os.path.dirname(self.phonopy_dir), 'primitive_cell')
        else:
            self.phonopy_dir = self.max_density_dir
        os.makedirs(self.primitive_cell_dir, exist_ok=True)
        CONTCAR_files = [f for f in os.listdir(self.phonopy_dir) if f.startswith('CONTCAR_')]
        # 改变工作目录，便于运行shell命令进行phonopy对称性检查和原胞与常规胞的生成
        os.chdir(self.phonopy_dir)
        for filename in CONTCAR_files:
            # 按顺序处理POSCAR文件，首先复制一份无数字后缀的POSCAR文件
            shutil.copy(f'{self.phonopy_dir}/{filename}', f'{self.phonopy_dir}/POSCAR')
            with open(f'{self.primitive_cell_dir}/phonopy.log', 'a') as log:
                # 使用phonopy模块处理POSCAR结构文件，获取对称化的原胞和常规胞。
                # 应用晶体的对称操作优化后的原胞可以最好地符合晶体的对称性，减少后续优化计算的复杂性。
                log.write(f'\nProcessing file: {filename}\n')
                result = subprocess.run(['phonopy', '--symmetry', 'POSCAR'], stderr=subprocess.STDOUT)
                log.write(f'Finished processing file: {filename} with return code: {result.returncode}\n')
            # 将phonopy生成的PPOSCAR（对称化原胞）和BPOSCAR（对称化常规胞）放到对应的文件夹中，并将文件名改回POSCAR_index
            shutil.move(f'{self.phonopy_dir}/PPOSCAR', f'{self.primitive_cell_dir}/{filename}')
        # 移除最后复制多出来的POSCAR文件和phonopy_symcells.yaml
        os.remove(f'{self.phonopy_dir}/phonopy_symcells.yaml')
        os.remove(f'{self.phonopy_dir}/POSCAR')
        os.remove(f'{self.phonopy_dir}/BPOSCAR')
        logging.info(f'The phonopy processing has been completed!!\nThe symmetrized primitive cells have been saved in POSCAR format to the primitive_cell folder.\nThe output content of phonopy has been saved to the phonopy.log file in the same directory.')


def log_and_time(func):
    """Decorator for recording log information and script runtime"""
    # 获取脚本所在目录, 在该目录下生成日志
    base_dir = os.path.dirname(__file__)
    script_name = os.path.basename(__file__) 
    log_file_path = os.path.join(base_dir, f'{script_name}_output.log')
    # 配置日志记录
    logging.basicConfig(
        filename = log_file_path,  # 日志文件名
        level = logging.INFO,  # 指定日志级别
        format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    )    
    def wrapper(*args, **kwargs):
        # 获取程序开始执行时的CPU时间和Wall Clock时间
        start_cpu, start_clock = time.process_time(), time.perf_counter()
        # 记录程序开始信息
        script_name = os.path.basename(__file__) 
        logging.info(f'Start running: {script_name}')
        # 调用实际的函数, 如果出现错误, 报错的同时也将错误信息记录到日志中
        result = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logging.error(f'Error occurred: {e}', exc_info=True)
            raise
        print('The script has run successfully, and the output content has been recorded in the output.log file in the same directory.')
        # 获取程序结束时的CPU时间和Wall Clock时间
        end_cpu, end_clock = time.process_time(), time.perf_counter()
        # 计算CPU时间和Wall Clock时间的差值
        cpu_time, wall_time = end_cpu-start_cpu, end_clock-start_clock
        # 记录程序结束信息
        logging.info(f'End running: {script_name}\nWall time: {wall_time:.4f} sec, CPU time: {cpu_time:.4f} sec\n')
        return result
    return wrapper


@ log_and_time
def main():
    # 分析处理机器学习势优化得到的CONTCAR文件
    result = CONTCAR_processing('mlp_results/combo_3/optimized')
    # 读取密度数据以及N5环是否独立存在，并将前n个最大密度的文件保存到max_density文件夹
    result.read_density_and_sort(n=30, N5_screen=True, count_N5=2)
    # 将max_density文件夹中的结构文件利用 phononpy 模块进行对称化处理，方便后续对于结构的查看，同时不影响晶胞性质
    result.phonopy_processing_max_density()

if __name__ == "__main__":
    main()
