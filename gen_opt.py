import os
import time
import glob
import yaml
import shutil
import logging
import argparse
import subprocess
from warnings import filterwarnings
from ase.io import read
from pyxtal import pyxtal
from pyxtal.msg import Comp_CompatibilityError, Symm_CompatibilityError
from dpdispatcher import Machine, Resources, Task, Submission

filterwarnings('ignore')

class CrystalGenerator:
    def __init__(self, work_dir:str, ion_numbers:list, species:list=[], conventional=False):
        """
        Initialize the class based on the provided ionic crystal composition structure files and corresponding composition numbers。
        """
        # 获取当前脚本的路径以及同路径下离子晶体组分的结构文件, 并将这一路径作为工作路径来避免可能的错误
        self.base_dir = work_dir
        os.chdir(self.base_dir)
        self.ion_numbers = ion_numbers
        if species:
            self.species = species
        else:
            self.species = glob.glob('*.gjf')
        self.species_dirs = []
        ion_atomss, species_atomss = [], []
        # 读取离子晶体各组分的原子数，并在日志文件中记录
        for ion, number in zip(self.species, self.ion_numbers):
            species_dir = os.path.join(self.base_dir, ion)
            self.species_dirs.append(species_dir)
            species_atom = len(read(species_dir))
            species_atomss.append(species_atom)
            species_atoms = species_atom * number
            ion_atomss.append(species_atoms)
        self.cell_atoms = sum(ion_atomss)
        logging.info(f'The components of ions {self.species} in the ionic crystal are {self.ion_numbers}')
        logging.info(f'The number of atoms for each ion is: {species_atomss}, and the total number of atoms is {self.cell_atoms}')
        # 创建脚本同路径下用于存放原胞和常规胞的文件夹
        self.POSCAR_dir = os.path.join(self.base_dir, '1_generated', 'POSCAR_Files')
        os.makedirs(self.POSCAR_dir, exist_ok=True)  # 如果目录不存在，则创建POSCAR_Files文件夹
        self.primitive_cell_dir = os.path.join(self.base_dir, '1_generated', 'primitive_cell')
        os.makedirs(self.primitive_cell_dir, exist_ok=True)
        # 根据conventional参数确定是否要保留conventional_cell结构文件
        self.conventional = conventional
        if self.conventional:
            self.conventional_cell_dir = os.path.join(self.base_dir, '1_generated', 'conventional_cell')
            os.makedirs(self.conventional_cell_dir, exist_ok=True)

        # 定义要检查的文件列表
        self.required_files = ['YZ_run_opt.py', '/workplace/yz/Test/yz_opt/ion_CSP/2_generation/template/model.pt']
        # 遍历文件列表，检查所需的每个文件是否存在
        for file in self.required_files:
            file_path = os.path.join(self.base_dir, file)
            if not os.path.exists(file_path):
                error_message = (f"\nWrong: The required file {file} does not exist in the current directory\nRequired files include: {self.required_files}.")
                logging.error(error_message)
                raise FileNotFoundError(error_message)
            else:
                # 准备dpdispatcher运行所需的文件，将其复制到primitive_cell文件夹中
                if file != '/workplace/yz/Test/yz_opt/ion_CSP/2_generation/template/model.pt':  # 由于model.pt势函数文件过大，不进行反复复制，防止复制过程出错
                    shutil.copy(file_path, self.primitive_cell_dir)
        logging.info('The necessary files are fully prepared.')

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

    def generate_structures(self, num_per_group=100, space_groups_limit=None):
        """
        Based on the provided ion species and corresponding numbers, use pyxtal to randomly generate ion crystal structures based on crystal space groups.
        """
        count_POSCAR = 0  # 用于给生成的POSCAR文件计数
        if space_groups_limit:  # 如果参数is_test为True，则只测试1到15的空间群，以节约测试时间
            space_groups = space_groups_limit
        else:  # 否则搜索所有的230个空间群
            space_groups = 230
        for space_group in range(1, space_groups+1): 
            logging.info(f'Space group: {space_group}')
            count_group = 0
            try:
                for i in range(num_per_group):  # 参数N确定对每个空间群所要生成的POSCAR结构文件个数
                    # 调用pyxtal类
                    pyxtal_structure = pyxtal(molecular=True)
                    # 根据阴阳离子结构文件与对应的配比以及空间群信息随机生成离子晶体，N取100以上
                    pyxtal_structure.from_random(dim=3, group=space_group, species=self.species_dirs, numIons=self.ion_numbers, conventional=False)
                    # 生成POSCAR_n文件
                    POSCAR_path = os.path.join(self.POSCAR_dir, f'POSCAR_{count_POSCAR}')
                    pyxtal_structure.to_file(POSCAR_path, fmt='poscar')
                    count_POSCAR += 1
                    count_group += 1
                logging.info(f' {count_group} POSCAR generated.')
            except (RuntimeError, Comp_CompatibilityError, Symm_CompatibilityError) as e:
                # 捕获对于某一空间群生成结构的运行时间过长、组成兼容性错误、对称性兼容性错误等异常，使结构生成能够完全进行而不中断
                logging.error(f'Generating structure error: {e}')
        logging.info(f'Using pyxtal.from_random, {count_POSCAR} ion crystal structures were randomly generated based on crystal space groups.')

    def phonopy_processing(self):
        """
        Use phonopy to check and generate symmetric primitive cells, reducing the complexity of subsequent optimization calculations, and preventing pyxtal.from_random from generating double proportioned supercells. 
        """
        POSCAR_file_index_pairs = self._sequentially_read_files(self.POSCAR_dir, prefix_name='POSCAR_')
        # 改变工作目录为POSCAR_Files，便于运行shell命令进行phonopy对称性检查和原胞与常规胞的生成
        os.chdir(self.POSCAR_dir)
        logging.info('Start running phonopy processing ...')
        for _, filename in POSCAR_file_index_pairs:
            # 按顺序处理POSCAR文件，首先复制一份无数字后缀的POSCAR文件
            shutil.copy(f'{self.POSCAR_dir}/{filename}', f'{self.POSCAR_dir}/POSCAR')
            with open(f'{self.primitive_cell_dir}/phonopy.log', 'a') as log:
                # 使用phonopy模块处理POSCAR结构文件，获取对称化的原胞和常规胞。
                # 应用晶体的对称操作优化后的原胞可以最好地符合晶体的对称性，减少后续优化计算的复杂性。
                log.write(f'\nProcessing file: {filename}\n')
                result = subprocess.run(['nohup', 'phonopy', '--symmetry', 'POSCAR'], stderr=subprocess.STDOUT)
                log.write(f'Finished processing file: {filename} with return code: {result.returncode}\n')
            # 将phonopy生成的PPOSCAR（对称化原胞）和BPOSCAR（对称化常规胞）放到对应的文件夹中，并将文件名改回POSCAR_index
            shutil.move(f'{self.POSCAR_dir}/PPOSCAR', f'{self.primitive_cell_dir}/{filename}')
            cell_atoms = len(read(f'{self.primitive_cell_dir}/{filename}'))
            # 检查生成的POSCAR中的原子数，如果不匹配则删除该POSCAR并在日志中记录
            if cell_atoms != self.cell_atoms:
                os.remove(f'{self.primitive_cell_dir}/{filename}')
                logging.error(f'The number of atoms in {filename} does not match!! Original: {self.cell_atoms} vs Generated {cell_atoms}')
            if self.conventional:
                shutil.move(f'{self.POSCAR_dir}/BPOSCAR', f'{self.conventional_cell_dir}/{filename}')
        # 移除最后复制多出来的POSCAR以及phonopy_symcells.yaml、nohup.out
        os.remove(f'{self.POSCAR_dir}/phonopy_symcells.yaml')
        os.remove(f'{self.POSCAR_dir}/POSCAR')
        if not self.conventional:
            os.remove(f'{self.POSCAR_dir}/BPOSCAR')
        logging.info(f'The phonopy processing has been completed!!\nThe symmetrized primitive cells have been saved in POSCAR format to the primitive_cell folder.\nThe output content of phonopy has been saved to the phonopy.log file in the same directory.')
        
    def prepare_and_submit(self, machine_file, resources_file, python_path='python', task_alloc=1):
        """
        Based on the dpdispatcher module, prepare and submit files for optimization on remote server or local machine.
        """
        # 调整工作目录，减少错误发生
        os.chdir(self.primitive_cell_dir)
        # 设置远程服务器上的python路径，读取machine.json和resources.json的参数
        machine = Machine.load_from_json(machine_file)
        resources = Resources.load_from_json(resources_file)
        # 由于dpdispatcher对于远程服务器以及本地运行的forward_common_files的默认存放位置不同，因此需要预先进行判断，从而不改动优化脚本
        machine_inform = machine.serialize()
        if machine_inform['context_type'] == 'SSHContext':
            # 如果调用远程服务器，则创建二级目录
            parent = 'data/'
            forward_common_files = [f'{self.base_dir}/model.pt']
        elif machine_inform['context_type'] == 'LocalContext':
            # 如果在本地运行作业，则只在后续创建一级目录
            parent = ''
            forward_common_files = []
        # 依次读取primitive_cell文件夹中的所有POSCAR文件和对应的序号
        primitive_cell_file_index_pairs = self._sequentially_read_files(self.primitive_cell_dir, prefix_name='POSCAR_')
        total_files = len(primitive_cell_file_index_pairs)
        logging.info(f'The total number of POSCAR files to be optimized: {total_files}')
        # 创建一个嵌套列表来存储每个GPU的任务并将文件平均依次分配给每个GPU
        # 例如：对于10个结构文件任务分发给4个GPU的情况，则4个GPU领到的任务分别[0, 4, 8], [1, 5, 9], [2, 6], [3, 7], 便于快速分辨GPU与作业的分配关系
        gpu_jobs = [[] for _ in range(task_alloc)]
        for index, _ in primitive_cell_file_index_pairs:
            gpu_index = index % task_alloc
            gpu_jobs[gpu_index].append(index)
        task_list = []
        for pop in range(task_alloc):
            remote_task_dir = f'{parent}pop{pop}'
            command = f'{python_path} YZ_run_opt.py > output_dp.log 2>&1'
            forward_files = ['YZ_run_opt.py']
            backward_files = ['output_dp.log'] 
            # 将YZ_run_opt.py和input.dat复制一份到task_dir下
            task_dir = os.path.join(self.primitive_cell_dir, f'{parent}pop{pop}')
            os.makedirs(task_dir, exist_ok=True)
            for file in forward_files:
                shutil.copyfile(f'{self.primitive_cell_dir}/{file}', f'{task_dir}/{file}')
            for job_i in gpu_jobs[pop]:
                # 将分配好的POSCAR文件添加到对应的上传文件中
                forward_files.append(f'POSCAR_{job_i}')
                # 每个POSCAR文件在优化后都取回对应的CONTCAR和OUTCAR输出文件
                backward_files.append(f'CONTCAR_{job_i}')
                backward_files.append(f'OUTCAR_{job_i}')
                shutil.copyfile(f'{self.primitive_cell_dir}/POSCAR_{job_i}', f'{task_dir}/POSCAR_{job_i}')
                shutil.copyfile(f'{self.primitive_cell_dir}/POSCAR_{job_i}', f'{task_dir}/ori_POSCAR_{job_i}')

            task = Task(
                command=command,
                task_work_path=remote_task_dir,
                forward_files=forward_files,
                backward_files=backward_files
            )
            task_list.append(task)

        submission = Submission(
            work_base=self.primitive_cell_dir,
            machine=machine,
            resources=resources,
            task_list=task_list,
            forward_common_files=forward_common_files
        )
        submission.run_submission()

        # 创建用于存放优化后文件的 mlp_optimized 目录   
        optimized_dir = os.path.join(self.base_dir, '2_mlp_optimized')
        os.makedirs(optimized_dir, exist_ok=True)
        for pop in range(task_alloc):
            # 从传回 primitive_cell 目录下的 pop 文件夹中将结果文件取到 mlp_optimized 目录
            task_dir = os.path.join(self.primitive_cell_dir, f'{parent}pop{pop}')
             # 按照给定的 POSCAR 结构文件按顺序读取 CONTCAR 和 OUTCAR 文件并复制
            task_file_index_pairs = self._sequentially_read_files(task_dir, prefix_name='POSCAR_')
            for index, _ in task_file_index_pairs:
                shutil.copyfile(f'{task_dir}/CONTCAR_{index}', f'{optimized_dir}/CONTCAR_{index}')
                shutil.copyfile(f'{task_dir}/OUTCAR_{index}', f'{optimized_dir}/OUTCAR_{index}')
        # 完成后删除不必要的运行文件并记录优化完成的信息
        os.remove(f'{self.primitive_cell_dir}/YZ_run_opt.py')
        os.remove(f'{self.base_dir}/model.pt')
        logging.info('Batch optimization completed!!!')

def log_and_time(func):
    """Decorator for recording log information and script runtime""" 
    def wrapper(work_dir, *args, **kwargs):
        # 获取脚本所在目录, 在该目录下生成日志
        script_name = os.path.basename(__file__) 
        log_file_path = os.path.join(work_dir, f'{script_name}_output.log')
        # 配置日志记录
        logging.basicConfig(
            filename = log_file_path,  # 日志文件名
            level = logging.INFO,  # 指定日志级别
            format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
        )   
        # 获取程序开始执行时的CPU时间和Wall Clock时间
        start_cpu, start_clock = time.process_time(), time.perf_counter()
        # 记录程序开始信息
        logging.info(f'Start running: {script_name}')
        # 调用实际的函数, 如果出现错误, 报错的同时也将错误信息记录到日志中
        result = None
        try:
            result = func(work_dir, *args, **kwargs)
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
def main(work_dir, gen_opt, config):
    generator = CrystalGenerator(work_dir, ion_numbers=config['gen_opt']['ion_numbers'])
    if gen_opt == 'gen':
        # 根据提供的离子与对应的配比，使用 pyxtal 基于晶体空间群进行离子晶体结构的随机生成。
        generator.generate_structures(num_per_group=config['gen_opt']['num_per_group'], 
                                      space_groups_limit=config['gen_opt']['space_groups_limit'])
        # 使用 phonopy 生成对称化的原胞另存于 primitive_cell 文件夹中，降低后续优化的复杂性，同时检查原子数以防止 pyxtal 生成双倍比例的超胞。
        generator.phonopy_processing()
    elif gen_opt == 'opt':
        # 基于 dpdispatcher 模块，在远程服务器上批量准备并提交输入文件，并在任务结束后回收机器学习势优化的输出文件 OUTCAR 与 CONTCAR
        generator.prepare_and_submit(machine_file=config['gen_opt']['machine_file'], 
                                     resources_file=config['gen_opt']['resources_file'],
                                     task_alloc=config['gen_opt']['task_alloc'])
    else:
        raise ValueError('Not an avalible "gen_opt" parameter, which should be "gen" or "opt"')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choice of generation or MLP optmization')
    parser.add_argument('work_dir', type=str, help='The working directory to run the script in')
    parser.add_argument('gen_opt', type=str, help='Choice of generation or MLP optmization')
    args = parser.parse_args()
    # 默认参数配置
    default_config = {
        'gen_opt': {
            'species': [],
            'ion_numbers': [2, 2, 2],
            'num_per_group': 500, 
            'space_groups_limit': 75,
            'machine_file': '/workplace/yz/Test/yz_opt/ion_CSP/0_server_config/6001_local/6001_local_machine.json',
            'resources_file': '/workplace/yz/Test/yz_opt/ion_CSP/0_server_config/6001_local/6001_local_resources.json',
            'task_alloc': 1
        }
    }
    # 尝试读取配置文件
    try:
        with open(os.path.join(args.work_dir, 'config.yaml'), 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        config = default_config
        print('config.yaml not found, using default config')
    main(args.work_dir, args.gen_opt, config)
