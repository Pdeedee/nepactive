import os
from nepactive import dlog
from nepactive.remote import Remotetask
import shutil
import subprocess
from ase.io import read, write, Trajectory
from ase.io.extxyz import write_extxyz
from glob import glob
from ase import Atoms
import random
from typing import List,Optional
from collections import Counter
import numpy as np
import json
from tqdm import tqdm
from ase import Atom
from pynep.calculate import NEP
from mattersim.forcefield import MatterSimCalculator
import random
import yaml
from math import ceil,floor
from torch.cuda import empty_cache
import itertools
from concurrent.futures import ThreadPoolExecutor
import re
import copy
from nepactive.stable import StableRun
from nepactive.template import npt_template,nphugo_template,nvt_template,msst_template,model_devi_template,nep_in_template,nphugo_pytemplate,nvt_pytemplate,continue_pytemplate
from nepactive.plt import gpumdplt,nep_plt,ase_plt
from nepactive import parse_yaml
from nepactive.force import force_main
from nepactive.tools import compute_volume_from_thermo,run_gpumd_task
import time
# from numba import jit


class RestartSignal(Exception):
    def __init__(self, restart_total_time = None):
        super().__init__()
        self.restart_total_time = restart_total_time

def process_id(value):
    # 检查是否为单一数字
    if re.match(r'^\d+$', value):
        return [int(value)]  # 如果是数字，返回列表形式的单个数字
    # 检查是否为有效的数字范围，如 "3-11"
    if re.match(r'^\d+-\d+$', value):
        start, end = value.split('-')
        start, end = int(start), int(end)
        if start > end:
            raise ValueError(f"范围的起始值不能大于结束值: {value}")
        return list(range(start, end + 1))  # 返回范围的数字列表
    # 如果不匹配数字或范围格式，抛出异常
    raise ValueError(f"无效的格式: {value}，必须是数字或者数字范围（如 '3-11'）")

def get_shortest_distance(atoms:Atoms,atom_index=None):
    distance_matrix = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distance_matrix, np.inf)
    min_index = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
    if atom_index is not None:
        atom_index.append(min_index)
    return np.min(distance_matrix)

def traj_write(atoms_list:Atoms, calculator):
    traj = Trajectory("out.traj", "w")
    for atoms in atoms_list:
        atoms._calc = calculator
        atoms.get_potential_energy()
        traj.write(atoms)    

def get_force(atoms:Atoms,calculator):
    atoms._calc=calculator
    return atoms.get_forces(),atoms.get_potential_energy()



# task = None
Maxlength = 70
def sepline(ch="-", sp="-"):
    r"""Seperate the output by '-'."""
    # if screen:
    #     print(ch.center(MaxLength, sp))
    # else:
    dlog.info(ch.center(Maxlength, sp))

def record_iter(record, ii, jj):
    with open(record, "a") as frec:
        frec.write("%d %d\n" % (ii, jj))

class Nepactive(object):
    def __init__(self,idata:dict):
        self.idata:dict = idata
        self.work_dir = os.getcwd()
        self.make_gpumd_task_first = True
        self.gpu_available = self.idata.get("gpu_available")

    def run(self):
        '''
        using different engines to make initial training data
        in.yaml need:
        init_engine: the engine to use for initial training data
        init_template_files: the template files to use for initial training data
        python_interpreter: the python interpreter to use for initial training data
        training_ratio: the ratio of training data to total data
        '''
        #change working directory

        engine = self.idata.get("ini_engine","ase")
        # label_engine = idata.get("label_engine","mattersim")
        work_dir = self.work_dir
        os.chdir(work_dir)
        os.makedirs("init",exist_ok=True)
        record = "record.nep"
        #init engine choice
        dlog.info(f"Initializing engine and make initial training data using {engine}")
        if not os.path.isfile(record):
            self.make_init_ase_run() 
            dlog.info("Extracting data from initial runs")
            self.make_data_extraction()
        self.make_loop_train()

    def make_init_ase_run(self):
        '''
        For the template file, the file name must be fixed form.
        Assumed that the working directory is already the correct directory.
        '''
        if_stable_run = self.idata.get("if_stable_run",False)
        os.makedirs("init",exist_ok=True)
        if os.path.exists("POSCAR"):
            shutil.copy("POSCAR","init")
        work_dir = f"{self.work_dir}/init"
        struc_dirs = []
        os.chdir(work_dir)
        #生成stablerun的任务，之后一块跑
        if if_stable_run:
            stable_run_data:dict = self.idata.get("stable")
            stable_run = StableRun(stable_run_data)
            stable_run.calculate_properties()
            for ii in range(stable_run.struc_num):
                os.chdir(work_dir)
                os.makedirs(f"struc.{ii:03d}",exist_ok=True)
                struc_dir = os.path.abspath(f"struc.{ii:03d}")
                struc_dirs.append(struc_dir)
                os.chdir(struc_dir)
                stable_run.make_preparations()

            original_make = stable_run_data.get("original_make",True)
            if original_make:
                os.chdir(work_dir)
                os.makedirs("original",exist_ok=True)
                struc_dir = os.path.abspath("original")
                struc_dirs.append(struc_dir)
                os.chdir(struc_dir)
                atoms = read(f"{self.work_dir}/POSCAR")
                os.makedirs("structure",exist_ok=True)
                write(f"structure/stable.pdb",atoms)
                stable_run.make_preparations()
            
        ase_ensemble_files = self.idata.get("ini_ase_ensemble_files")
        if ase_ensemble_files:
            ase_ensemble_files:list[str] = [os.path.abspath(path) for path in ase_ensemble_files]

        python_interpreter:str = self.idata.get("python_interpreter")
        processes = []
        self.pot_file:str = self.idata.get("pot_file")
        pot_file =self.pot_file
        self.gpu_available = self.idata.get("gpu_available")
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        # Make initial training task directories
        if ase_ensemble_files:
            for index, file in enumerate(ase_ensemble_files):
                task_name = f"task.{index:06d}"
                # Ensure the task directory is created
                task_dir = os.path.join(work_dir, task_name)
                os.makedirs(task_dir, exist_ok=True)  
                # Create the task directory, if it doesn't exist
                os.system(f"ln -snf {pot_file} {task_dir}/model.pth")
                os.system(f"ln -snf {file} {task_dir}")
        
        os.chdir(work_dir)
        task_dirs = glob("./**/task.*", recursive=True)
        if task_dirs:
            task_dirs = [os.path.abspath(path) for path in task_dirs]

        for index,task_dir in enumerate(task_dirs):
            os.chdir(task_dir)
            if os.path.exists("task_finished"):
                dlog.warning(f"{task_dir} has already been finished, skip it")
                continue
            # basename = os.path.basename(file)
            basename = "ensemble.py"
            gpu_id = self.gpu_available[index%len(self.gpu_available)]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # Open subprocess and redirect stdout and stderr to a log file
            log_file = os.path.join(task_dir, 'log')  # Log file path
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    [python_interpreter, basename], 
                    stdout=log, 
                    stderr=subprocess.STDOUT,  # Combine stderr with stdout
                    env = env
                )
                processes.append((process, task_dir))  # Store the process and log file

        # Wait for all subprocesses to complete and check for errors
        for process, task_dir in processes:
            process.wait()  # Wait for the process to complete
            # Check for errors using the return code
            if process.returncode != 0:
                dlog.error(f"Process failed. Check the log at: {task_dir}/log")
                raise RuntimeError(f"Process failed. Check the log at: {task_dir}/log")
            else:
                os.chdir(task_dir)
                ase_plt()
                os.system(f"touch {task_dir}/task_finished")
                dlog.info(f"Process completed successfully. Log saved at: {task_dir}/log")

        # All scripts executed, proceed to the next step
        dlog.info("Initial training data generated")

    def make_data_extraction(self):
        '''
        extract data from initial runs, turn it into the gpumd format
        '''
        train:List[Atoms]=[]
        test:List[Atoms]=[]
        training_ratio:float = self.idata.get("training_ratio", 0.8)
        init_frames:int = self.idata.get("ini_frames", 100)
        # work_dir = f"{work_dir}/init"
        os.chdir(self.work_dir)

        ####检查
        fs = glob("init/**/task.*/*.traj",recursive=True)
        dlog.info(f"Found {len(fs)} files to extract frames from.")
        if not fs:
            raise ValueError("No files found to extract frames from.")
        # may report error due to the format not matching
        atoms = []
        
        # average sampling
        fnumber= len(fs)
        needed_frames = init_frames/fnumber
        
        for f in fs:
            atom = read(f,index=":")
            if len(atom) < needed_frames:
                dlog.warning(f"Not enough frames in {f} to extract {needed_frames} frames.")
                # raise ValueError(f"Not enough frames in {f} to extract {needed_frames} frames.")
            interval = max(1, floor(len(atom) / needed_frames))  # 确保 interval 至少为 1
            atom = atom[::interval]
            atoms.extend(atom)

        assert atoms is not None

        if len(atoms) < init_frames:
            dlog.warning(f"Not enough frames to extract {init_frames} frames.")
        elif len(atoms) > init_frames:
            # dlog.warning(f"Too many frames to extract {init_frames} frames.")
            # raise ValueError(f"Not enough frames to extract {init_frames} frames.")
            atoms = atoms[:init_frames]
        else:
            dlog.info(f"Extracted {init_frames} frames from {fnumber} files.")

        write_extxyz("init/init.xyz", atoms)
        for i in range(len(atoms)):
            rand=random.random()
            if rand <= training_ratio:
                train.append(atoms[i])
            elif rand > training_ratio:
                test.append(atoms[i])
            else:
                dlog.warning(f"{atoms[i]}failed to be classified")
        write_extxyz("init/train.xyz", train)
        write_extxyz("init/test.xyz", test)
        dlog.info("Initial training data extracted")

    def make_loop_train(self):
        '''
        make loop training task
        '''
        
        os.chdir(self.work_dir)

        record = "record.nep"
        iter_rec = [0, -1]
        if os.path.isfile(record):
            with open(record) as frec:
                for line in frec:
                    iter_rec = [int(x) for x in line.split()]
        cont = True
        self.ii = -1
        numb_task = 9
        max_tasks = 10000
        self.restart_total_time = None

        while True:
            self.ii += 1
            if self.ii < iter_rec[0]:
                continue
            self.iter_dir = os.path.abspath(os.path.join(self.work_dir,f"iter.{self.ii:06d}")) #the work path has been changed
            os.makedirs(self.iter_dir, exist_ok=True)
            iter_name = f"iter.{self.ii:06d}"
            self.restart_gpumd = False

            for jj in range(numb_task):
                if not self.restart_gpumd:
                    self.jj = jj
                yaml_synchro = self.idata.get("yaml_synchro", False)
                if yaml_synchro:
                    dlog.info(f"yaml_synchro is True, reread the in.yaml from {self.work_dir}/in.yaml")
                    self.idata:dict = parse_yaml(f"{self.work_dir}/in.yaml")
                if self.ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1] and not self.restart_gpumd:
                    continue
                task_name = "task %02d" % jj
                sepline(f"{iter_name} {task_name}", "-")
                if jj == 0:
                    self.make_nep_train()
                elif jj == 1:
                    self.run_nep_train()
                elif jj == 2:
                    self.post_nep_train()
                elif jj == 3:
                    self.make_model_devi()
                elif jj == 4:
                    self.run_model_devi()
                elif jj == 5:
                    self.post_gpumd_run()
                elif jj == 6:
                    self.make_label_task()
                elif jj == 7:
                    self.run_label_task()
                elif jj == 8:
                    self.post_label_task()
                else:
                    raise RuntimeError("unknown task %d, something wrong" % jj)
                
                os.chdir(self.work_dir)
                record_iter(record, self.ii, jj)

    def shock_vel_test(self):
        work_dir = os.path.abspath(os.path.join(self.iter_dir, "03.shock"))
        os.makedirs(work_dir, exist_ok=True)
        atoms = read(f"{self.work_dir}/POSCAR")
        write(f"{work_dir}/POSCAR", atoms)
        os.system(f"ln -snf {self.work_dir}/init/properties.txt {work_dir}/properties.txt")
        os.chdir(work_dir)
        stable_data = self.idata.get("stable", None)
        assert stable_data is not None, "stable data is None"
        nep_file = os.path.join(self.iter_dir, "00.nep/task.000000/nep.txt")
        stable_data["nep"] = nep_file
        stable_data["pot"] = "nep"
        original_make = stable_data.get("original_make", False)
        if original_make:
            structure_files = [os.path.abspath(self.idata.get("structure_files")[0])]
        else:
            structure_files = []
        final_xyzs = glob(f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd/task.[0-9][0-9][0-9][0-9][0-9][0-9]/final.xyz")
        final_xyzs.sort()
        final_xyz = final_xyzs[-1]
        structure_files.append(final_xyz)
        stable_data["structure_files"] = structure_files
        
        analyze_range = stable_data.get("analyze_range", None)
        if analyze_range is None:
            analyze_range = self.idata.get("analyze_range", [0.5,1])
            stable_data["analyze_range"] = analyze_range
            dlog.info(f"analyze_range in stable_data is None, set to {analyze_range} in idata")

        stable_task = StableRun(stable_data)
        stable_task.run()

    def run_nep_train(self):
        '''
        run nep training
        '''
        train_steps = self.idata.get("train_steps", 5000)
        work_dir = os.path.abspath(os.path.join(self.iter_dir, "00.nep"))
        pot_num = self.idata.get("pot_num", 4)
        pot_inherit:bool = self.idata.get("pot_inherit", True)
        # nep_template = os.path.abspath(self.idata.get("nep_template"))
        processes = []
        if not pot_inherit:
            dlog.info(f"{os.getcwd()}")
            dlog.info(f"pot_inherit is false, will remove old task files {work_dir}/task*")
            os.system(f"rm -r {work_dir}/task.*")
        for jj in range(pot_num):
            #ensure the work_dir is the absolute path
            task_dir = os.path.join(work_dir, f"task.{jj:06d}")
            os.makedirs(task_dir,exist_ok=True)
            #     absworkdir/iter.000000/00.nep/task.000000/
            os.chdir(task_dir)
            #preparation files
            if not os.path.isfile("train.xyz"):
                os.symlink("../dataset/train.xyz","train.xyz")
            if not os.path.isfile("test.xyz"):
                os.symlink("../dataset/test.xyz","test.xyz")
            if not os.path.isfile("nep.in"):
                nep_in_header = self.idata.get("nep_in_header", "type 4 H C N O")
                if self.ii == 0:
                    ini_train_steps = self.idata.get("ini_train_steps", 10000)
                    nep_in = nep_in_template.format(train_steps=ini_train_steps,nep_in_header=nep_in_header)
                else:
                    nep_in = nep_in_template.format(train_steps=train_steps,nep_in_header=nep_in_header)
                with open("nep.in", "w") as f:
                    f.write(nep_in)
                # os.symlink(nep_template, "nep.in")
            if pot_inherit and self.ii > 0:
                nep_restart = f"{self.work_dir}/iter.{self.ii-1:06d}/00.nep/task.{jj:06d}/nep.restart"
                dlog.info(f"pot_inherit is true, will copy nep.restart from {nep_restart}")
                shutil.copy(nep_restart, "nep.restart")
                # exit()
            log_file = os.path.join(task_dir, 'log')  # Log file path
            env = os.environ.copy()
            gpu_id = self.gpu_available[jj%len(self.gpu_available)]
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # env['CUDA_VISIBLE_DEVICES'] = str(jj)
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    ["nep"],  # 程序名
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env  # 使用修改后的环境
                )
                processes.append((process, log_file))
            
            self.gpu_numbers = len(self.idata.get("gpu_available"))            
            if (jj+1)%self.gpu_numbers == 0:
                dlog.info(f"jobs submitted, checking status of {self.jj}")
                for process, log_file in processes:
                    process.wait()  # Wait for all processes to complete
                    # Check for errors using the return code
                    if process.returncode != 0:
                        dlog.error(f"Process failed. Check the log at: {log_file}")
                        raise RuntimeError(f"One or more processes failed. Check the log file:({log_file}) for details.")
                    else:
                        dlog.info(f"Process completed successfully. Log saved at: {log_file}")

    def post_label_task(self):
        '''
        '''
        test_interval = self.idata.get("shock_test_interval", 1)
        test_begin_step = self.idata.get("shock_test_begin_step", 400000)

        self.run_steps = int(np.loadtxt(f"{self.work_dir}/steps.txt",ndmin=1,encoding="utf-8")[-1])
        if self.run_steps > test_begin_step and self.ii%test_interval == 0:
            dlog.info(f"run_steps is {self.run_steps}, will run shock velocity test")
            self.shock_vel_test()
    
    def shock(self):
        record = "record.nep"
        iter_rec = [0, -1]
        if os.path.isfile(record):
            with open(record) as frec:
                for line in frec:
                    iter_rec = [int(x) for x in line.split()]
        else:
            dlog.error(f"record file {record} not found")
            raise ValueError(f"record file {record} not found")
        
        if iter_rec[1] >=4:
            iter = iter_rec[0]
        else:
            iter = iter_rec[0] - 1
        self.ii = iter
        self.iter_dir = os.path.abspath(os.path.join(self.work_dir,f"iter.{iter:06d}"))
        self.shock_vel_test() 

        

    def post_nep_train(self):
        '''
        post nep training
        '''
        os.chdir(f"{self.work_dir}/iter.{self.ii:06d}/00.nep")
        tasks = glob("task.*")
        nep_plot = self.idata.get("nep_plot", True)
        for task in tasks:
            if nep_plot:
                os.chdir(f"{self.work_dir}/iter.{self.ii:06d}/00.nep")
                os.chdir(task)
                nep_plt()
                nep_plt(testplt=False)

    def make_nep_train(self):
        '''
        Train nep. 
        '''
        #ensure the work_dir is the absolute pathk
        global_work_dir = os.path.abspath(self.work_dir)
        work_dir = os.path.abspath(os.path.join(self.iter_dir, "00.nep"))
        pot_num = self.idata.get("pot_num", 4)
        init:bool = self.idata.get("init", True)

        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        #     absworkdir/iter.000000/00.nep/


        #preparation files
        if pot_num > 4:
            raise ValueError(f"pot_num should be no bigger than 4, and it is now {pot_num}")
        
        #make training data preparation
        os.makedirs("dataset", exist_ok=True)
        def merge_files(input_files, output_file):
            command = ['cat'] + input_files  # 适用于类Unix系统
            with open(output_file, 'wb') as outfile:
                subprocess.run(command, stdout=outfile)
        files = []
        testfiles = []
        # 注意每一代有没有划分训练集和测试集
        if init == True:
            print(f"{os.path.isfile(os.path.join(global_work_dir, 'init/train.xyz'))}")
            # 直接调用 extend 方法，不要尝试将其结果赋值
            files.extend(glob(os.path.join(global_work_dir, "init/train.xyz")))
            testfiles.extend(glob(os.path.join(global_work_dir, "init/test.xyz")))

            # 检查文件列表是否为空
            # if not files:
            #     raise ValueError("No files found to merge.")
        """等于之后反而出错了"""

        if self.ii > 0:
            for iii in range(self.ii):
                newtrainfile = glob(f"../../iter.{iii:06d}/02.label/iter_train.xyz")
                files.extend(newtrainfile)
                newtestfile = glob(f"../../iter.{iii:06d}/02.label/iter_test.xyz")
                testfiles.extend(newtestfile)
                if (not newtrainfile) or (not newtestfile):
                    dlog.warning(f"iter.{iii:06d} has no training or test data")
        if not files:
            raise ValueError("No files found to merge.")
            # dlog.error("No files found to merge.")
        merge_files(files, "dataset/train.xyz")
        merge_files(testfiles, "dataset/test.xyz")
        self.gpu_available = self.idata.get("gpu_available")



        # work_dir = os.path.abspath(f"{work_dir}/task.{ii:06d}")

    def make_model_devi(self):
        '''
        run gpumd, this function is referenced by make_loop_train
        '''

        model_devi = self.get_model_devi()

        if self.make_gpumd_task_first:
            # 日志记录
            dlog.info(f"make_gpumd_task_first is true, will backup old task directory {self.work_dir}/iter.{self.ii:06d}/01.gpumd")

            # 查找已有的备份文件夹
            bak_files = glob(f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd.bak.*")
            if bak_files:
                # 提取最大后缀数字
                suffixes = [int(os.path.basename(f).split('.')[-1]) for f in bak_files]
                new_suffix = max(suffixes) + 1
            else:
                # 如果没有备份文件夹，默认后缀为 0
                new_suffix = 0

            # 构造新备份路径
            src_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
            dst_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd.bak.{new_suffix}"

            # 重命名文件夹
            if os.path.exists(src_dir):
                shutil.move(src_dir, dst_dir)
                dlog.info(f"Backup completed: {src_dir} -> {dst_dir}")
            else:
                dlog.warning(f"Source directory does not exist: {src_dir}")


        work_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
        structure_files:list = self.idata.get("structure_files")
        structure_prefix = self.idata.get("structure_prefix",self.work_dir)
        time_step_general = self.idata.get("model_devi_time_step", None)
        structure_files = [os.path.join(structure_prefix,structure_file) for structure_file in structure_files]
        nep_file = self.idata.get("nep_file","../../00.nep/task.000000/nep.txt")
        needed_frames = self.idata.get("needed_frames",10000)
        self.run_steps_factor = self.idata.get("run_steps_factor",1.25)
        if self.ii == 0:
            run_steps = self.idata.get("ini_run_steps",100000)
            self.run_steps = run_steps
        else:
            all_run_steps = np.loadtxt(f"{self.work_dir}/steps.txt",ndmin=1,encoding="utf-8")
            old_run_steps = all_run_steps[-1]
            run_steps = int(self.run_steps_factor*int(old_run_steps))
            if len(all_run_steps) > 1:
                if all_run_steps[-2] > all_run_steps[-1]:
                    dlog.warning(f"The older run_steps is {old_run_steps}, the new run_steps is {all_run_steps[-1]},and the older one is bigger")
            self.run_steps = run_steps
            dlog.info(f"run_steps is {run_steps}")
        # if self.run_steps < 10000:

        if os.path.exists(f"{self.work_dir}/init/properties.txt"):
            property = np.loadtxt(f"{self.work_dir}/init/properties.txt",encoding="utf-8")
            e0 = [property[1]]
            p0 = [property[2]]
            v0 = [property[3]]
        else:
            e0 = [0]
            p0 = [0]
            v0 = [0]

        self.total_time, self.model_devi_task_numbers = Nepactive.make_gpumd_task(model_devi=model_devi, structure_files=structure_files, needed_frames=needed_frames,time_step_general=time_step_general,
                                                                                   work_dir=work_dir,nep_file=nep_file, run_steps=run_steps,e0=e0, p0=p0,v0 =v0)

    @classmethod
    def make_gpumd_task(cls, model_devi:dict, structure_files, needed_frames=10000, time_step_general=0.2, work_dir:str=None, 
                        nep_file:str=None, run_steps:int=20000, e0=[0], p0=[0], v0=[0]):
        if not work_dir:
            work_dir = os.getcwd()
        if not nep_file:
            nep_file = f"{work_dir}/nep.txt"


        if not (e0 and p0 and v0):
            e0 = model_devi.get("e0",None)
            p0 = model_devi.get("p0",None)
            v0 = model_devi.get("v0",None)

        assert e0 and p0 and v0, "e0, p0, v0 are not empty set"

        task_dicts = []
        ensembles = model_devi.get("ensembles")
        for ensemble_index,ensemble in enumerate(ensembles):
            structure_id = model_devi.get("structure_id")[ensemble_index]
            all_dict = {}
            assert structure_files is not None
            structure = [structure_files[ii] for ii in structure_id] #####################################################
            all_dict["structure"] = structure
            time_step = model_devi.get("time_step")

            if not time_step:
                time_step = time_step_general

            assert all([structure,time_step,run_steps]), "有变量为空"
            # task_dict = {}

            if ensemble in ["nvt", "npt", "nphugo"]:
                temperature = model_devi.get("temperature") #
                if ensemble in ["npt", "nvt"]:
                    assert temperature is not None
                    all_dict["temperature"] = temperature
                if ensemble in ["npt", "nphugo"]:
                    pressure = model_devi.get("pressure") #
                    all_dict["pressure"] = pressure
                    assert pressure is not None
                if ensemble == "nphugo":
                    all_dict["e0"] = e0
                    all_dict["p0"] = p0
                    all_dict["v0"] = v0
            elif ensemble == "msst":
                v_shock:list = model_devi.get("v_shock",[9.0])       #
                qmass:list = model_devi.get("qmass",[100000])           #
                viscosity:list = model_devi.get("viscosity",[10])   #
                shock_direction = model_devi.get("shock_direction","x")
                all_dict["v_shock"] = v_shock
                all_dict["qmass"] = qmass
                all_dict["viscosity"] = viscosity
                all_dict["shock_direction"] = shock_direction
            else:
                raise NotImplementedError
            dlog.info(f"all_dict:{all_dict}")
            assert all(v not in [None, '', [], {}, set()] for v in all_dict.values())
            # task_para = []
            # combo_numbers = 0
            for combo in itertools.product(*all_dict.values()):
                # 将每一组合生成字典，字典的键是列表的变量名，值是组合中的对应元素
                combo_dict = {keys: combo[index] for index, keys in enumerate(all_dict.keys())}
                task_dicts.append((ensemble,combo_dict))
            dlog.info(f"{all_dict.keys()} generate {len(task_dicts)} tasks")
            frames_pertask = needed_frames/len(task_dicts)
            dump_freq = max(1,floor(run_steps/frames_pertask))

            model_devi_task_numbers = len(task_dicts)
            replicate_cell = model_devi.get("replicate_cell","1 1 1")
            # nep_file = self.idata.get("nep_file","../../00.nep/task.000000/nep.txt")

        index = 0
        for ensemble,task in task_dicts:
            if ensemble == "msst":
                assert run_steps > 20000
                text = msst_template.format(time_step = time_step,run_steps = run_steps-20000,dump_freq = dump_freq, replicate_cell = replicate_cell, **task)
            elif ensemble == "nvt":
                text = nvt_template.format(time_step = time_step,run_steps = run_steps,dump_freq = dump_freq, replicate_cell = replicate_cell, **task)
            elif ensemble == "npt":
                text = npt_template.format(time_step = time_step,run_steps = run_steps,dump_freq = dump_freq, replicate_cell = replicate_cell, **task)
            elif ensemble == "nphugo":
                assert run_steps > 20000
                text = nphugo_template.format(
                    time_step = time_step,
                    run_steps = run_steps-20000,
                    dump_freq = dump_freq,
                    replicate_cell = replicate_cell,
                    **task
                )
                # dlog.info(f"nphugo task:{task},npugo text:{text}")
            else:
                raise NotImplementedError(f"The ensemble {ensemble} is not supported")
            task_dir = f"{work_dir}/task.{index:06d}"
            file = f"{task_dir}/run.in"
            os.makedirs(task_dir,exist_ok=True)
            os.chdir(task_dir)
            structure:str = task["structure"]
            if not structure.endswith("xyz"):
                atom = read(structure)
                atom = write("POSCAR",atom)
                atom = read("POSCAR")
                write_extxyz(f"model.xyz",atom)####################
            else:
                shutil.copy(structure,f"model.xyz")
            with open(file,mode='w') as f:
                f.write(text)
            if not os.path.isfile("nep.txt"):
                os.symlink(nep_file, "nep.txt")
            index += 1
        total_time = run_steps*time_step

        # dlog.info("generate gpumd task done")
        return total_time, model_devi_task_numbers

    def run_model_devi(self):
        # dlog.info("entering the run_gpumd_task")
        try:
            self.dump_freq
        except AttributeError:
            # 捕获变量不存在时的错误
            self.make_gpumd_task_first = False
            self.make_model_devi()
            self.make_gpumd_task_first = True
            dlog.info("remake the gpumd task")
        model_devi_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
        os.chdir(model_devi_dir)
        gpu_available = self.idata.get("gpu_available")
        self.task_per_gpu = self.idata.get("task_per_gpu")

        Nepactive.run_gpumd_task(work_dir=model_devi_dir, gpu_available=gpu_available, task_per_gpu=self.task_per_gpu)
        # self.write_steps()

    @classmethod
    def run_gpumd_task(cls,work_dir:str=None,gpu_available:List[int]=None,task_per_gpu:int=1):
        run_gpumd_task(work_dir=work_dir, gpu_available=gpu_available, task_per_gpu=task_per_gpu)

        
    def get_model_devi(self):
        '''
        get the model deviation from the gpumd run
        '''
        # dlog.info(f"self.idata:{self.idata}")
        model_devi_general:list[dict] = self.idata.get("model_devi_general", None)
        # dlog.info(f"model_devi_general:{model_devi_general}")
        if os.path.exists(os.path.join(self.work_dir, "model_devi_general.txt")):
            file = os.path.join(self.work_dir, "model_devi_general.txt")
            with open(file, mode='r') as f:
                model_devi_general = float(f.read()[-1])
            self.model_devi_general_id = np.loadtxt(os.path.join(self.work_dir, "model_devi_general_id.txt"), dtype=int)[-1]
        else:
            self.model_devi_general_id = 0
        
        if self.model_devi_general_id >= len(model_devi_general):
            dlog.info(f"finished")
            exit()
        
        model_devi = model_devi_general[self.model_devi_general_id] 
        # find_id = False
        # run_steps_length = 0
        # if not iteration:
            # iteration = self.ii
        # dlog.info(f"iteration:{iteration}")
        # for ii, model_devi in enumerate(model_devi_general):
            # model_devi_id = process_id(model_devi.get("id", None))
            # each_run_steps = model_devi.get("each_run_steps", [0])
            # now_run_steps_length = len(model_devi_id)
            # if iteration in model_devi_id:
            #     find_id = True
            #     model_devi = model_devi_general[ii]
            #     runsteps_id =  iteration - run_steps_length

            #     if runsteps_id < 0 or runsteps_id >= len(each_run_steps):
            #         runsteps_id = -1
            #         dlog.info("the each_runsteps is not enough, will use the last one")
            #     model_devi["run_steps"] = each_run_steps[runsteps_id]
            #     dlog.info(f"{ii}th model_devi_general has run_steps:{model_devi['run_steps']}")
            #     break
            # run_steps_length += now_run_steps_length   #skip the run_steps_length

        # if not find_id:
            # dlog.info(f"the ii:{iteration} is not in model_devi_general, will use the last one")
            # dlog.error(f"not finding the {iteration}th task settings in model_devi_general, will end the job")
            # raise RuntimeError(f"not finding the {iteration}th task settings in model_devi_general, will end the job")
        return model_devi

    def post_gpumd_run(self):
        '''
        extract the information from gpumd run, this function is referenced by make_loop_train
        '''
        try:
            self.total_time
        except AttributeError:
            # 捕获变量不存在时的错误
            self.make_gpumd_task_first = False
            self.make_model_devi()
            self.make_gpumd_task_first = True
            dlog.info("remake the gpumd task")

        nep_dir = os.path.join(self.iter_dir, "00.nep")
        plot = self.idata.get("gpumd_plt",True)

        if plot:
            model_devi_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
            os.chdir(model_devi_dir)
            task_dirs = [os.path.join(model_devi_dir,task) for task in glob("task*")]
            task_dirs.sort()
            # dlog.info(f"plotting {len(task_dirs)} tasks")
            dlog.info(f"plotting {len(task_dirs)} tasks")
            for task_dir in task_dirs:
                #ensure the work_dir is the absolute path
                os.chdir(task_dir)
                self.time_step = self.idata.get("time_step")
                try:
                    gpumdplt(self.total_time,self.time_step)
                except Exception as e:
                    dlog.error(f"plotting {task_dir} failed, error message:{e}")
                    raise RuntimeError(f"plotting {task_dir} failed, error message:{e}")

        self.gpumd_dir = os.path.join(self.iter_dir, "01.gpumd")
        os.chdir(self.gpumd_dir)

        model_devi:dict = self.get_model_devi()
        self.max_candidate = self.idata.get("max_candidate",1000)
        continue_from_old = self.idata.get("continue_from_old", False)

        sample_method = self.idata.get("sample_method", "relative")
        
        threshold = model_devi.get("uncertainty_threshold",None)
        if not threshold:
            threshold = self.idata.get("tuncertainty_threshold",[0.2,0.5])
        mode = self.idata.get("uncertainty_mode", "mean")
        energy_threshold = self.idata.get("energy_threshold", None)
        level = self.idata.get("uncertainty_level", 1)
        frame_properties = None
        atom_lists = None
        thermo_averages = None
        dlog.info(f"-----start analysis the trajectory-----"
                  f"threshold:{threshold},energy_threshold:{energy_threshold},mode:{mode},level:{level},sample_method:{sample_method}"
                  f"continue_from_old:{continue_from_old},max_candidate:{self.max_candidate},uncertainty_threshold:{threshold}")
        
        task_dirs.sort()
        failed_row_indices = []
        frame_properties = []
        used_frame_properties = []
        atom_lists = []
        candidate_indices_list = []
        accurate_len = 0
        candidate_len = 0
        max_candidate_per_task = self.max_candidate // len(task_dirs)
        candidate_list = []
        label_dir = os.path.join(self.iter_dir, "02.label")
        fmt = "%14d %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f"
        header = f"{'indices':>14} {'time':^14} {'relative_error':^14} {'energy_error':^14} {'shortest_d':^14} {'temperature':^14} {'potential':^14} {'pressure':^14} {'volume':^14}"
        os.makedirs(label_dir, exist_ok=True)
        if os.path.exists(os.path.join(label_dir, "candidate.xyz")):
            os.remove(os.path.join(label_dir, "candidate.xyz"))

        # candidate_max_temp = 7000
        candidate_max_temp = self.idata.get("max_temp", 7000)
        candidate_shortest_d = self.idata.get("shortest_d",0.6)
        analyze_range = self.idata.get("analyze_range", [0.5, 1.0])
        dlog.info(f"max_temp:{candidate_max_temp},shortest_d:{candidate_shortest_d},analyze_range:{analyze_range}")
        for ii,task_dir in enumerate(task_dirs):
            os.chdir(task_dir)
            dlog.info(f"processing task {ii}")
            atoms_list, frame_property, failed_row_index = Nepactive.relative_force_error(total_time=self.total_time, nep_dir=nep_dir, mode=mode,
                                                                                          level=level, allowed_max_temp = candidate_max_temp, allowed_shortest_distance = candidate_shortest_d)
            
            # used_frame_property = frame_property[:failed_row_index]

            # 保存最后一帧的坐标
            write_extxyz(os.path.join(task_dir, "final.xyz"), atoms_list[-1])
            dlog.info(f"the last frame is saved in {task_dir}/final.xyz")

            failed_row_indices.append(failed_row_index)

            thermo_average = np.average(frame_property[int(analyze_range[0] * len(frame_property)):int(analyze_range[1] * len(frame_property))],axis=0)[1:][np.newaxis,:]
            if thermo_averages is None:
                thermo_averages = thermo_average
            else:
                thermo_averages = np.vstack((thermo_averages, thermo_average))  # 垂直堆叠

            atom_lists.append(atoms_list)
            frame_properties.append(frame_property)

            candidate_condition = (((frame_property[:, 1] >= threshold[0]) & (frame_property[:, 1] <= threshold[1]))  
                        | (frame_property[:, 2] > energy_threshold)) & (frame_property[:,3] > candidate_shortest_d) & (frame_property[:,4] < candidate_max_temp)
            candidate_indices = np.where(candidate_condition)[0]
            candidate_indices_list.append(candidate_indices)
            filtered_rows = frame_property[candidate_indices]
            filtered_rows_with_indices = np.column_stack((candidate_indices, filtered_rows))
            
            accurate_condition = (frame_property[:, 1] < threshold[0]) & (frame_property[:, 2] < energy_threshold)
            accurate_len += len(np.where(accurate_condition)[0]) 
            candidate_len += len(candidate_indices) / len(frame_properties)

            # 按照不确定性从大到小排序筛选最大容许数
            if candidate_len > max_candidate_per_task:
                sorted_rows = filtered_rows_with_indices[filtered_rows_with_indices[:, 2].argsort()[::-1]]
                selected_rows = sorted_rows[:max_candidate_per_task]
            else:
                selected_rows = filtered_rows_with_indices
            
            # 按照indices从小到大排序
            final_sorted_rows:list = selected_rows[selected_rows[:, 0].argsort()]
            candidate_list:List[Atoms] = [atom_lists[ii][i] for i in final_sorted_rows[:, 0].astype(int)]
            # candidate_list:List[Atoms] = candidate_list.extend(candidate_atoms_list)
            with open(os.path.join(label_dir, "candidate.xyz"), "a") as f:
                write_extxyz(f, candidate_list)
            np.savetxt("candidate.txt", final_sorted_rows, fmt = fmt, header = header,comments=f"_{ii}_")
            dlog.info(f"the {ii}th task candidate has been written in {label_dir}/candidate.txt")


        ###### frameproperty的类型变了
        erliest_failed_allow_ratio = 0.8
        failed_row_indices = np.array(failed_row_indices)


        accurate_ratio = accurate_len/(len(frame_property)*len(frame_properties))
        candidate_ratio = candidate_len/(len(frame_property)*len(frame_properties))
        failed_ratio = 1 - accurate_ratio - candidate_ratio
        dlog.info(f"failed ratio: {failed_ratio}, candidate ratio: {candidate_ratio}, accurate ratio: {accurate_ratio}")

        with open(f'{self.work_dir}/thermo.txt', 'a') as f:
            np.savetxt(f, thermo_averages, fmt="%24.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f", 
                    header=f"{'relative_error':^14} {'energy_error':^14} {'shortest_d':^14} {'temperature':^14} {'potential':^14} {'pressure':^14} {'volume':^14}",
                    comments=f"#iter.{self.ii:06d}")

        if np.any(failed_row_indices < erliest_failed_allow_ratio*len(frame_property)):
            dlog.info(f"frames are failed from some index, the indices are {failed_row_indices}")
            min_failed_index = np.min(np.array(failed_row_indices))
            # self.run_steps = np.loadtxt(f"{self.work_dir}/steps.txt",ndmin=1)[-1]
            # 筛选出真正失败的 index 和对应的 task_dirs
            mask = failed_row_indices != len(frame_property)
            real_failed_row_indices = failed_row_indices[mask]
            failed_task_dirs = np.array(task_dirs)[mask]

            self.run_steps = max(22000,int(self.run_steps*min_failed_index/len(frame_property)))

            dlog.info(f"the failed task dirs are {failed_task_dirs} and indices are {real_failed_row_indices}")
            self.handle_bad_job(failed_row_indices=real_failed_row_indices,failed_task_dirs=failed_task_dirs)
            self.write_steps()
            return
        else:
            dlog.info(f"all frames are successful")

        os.chdir(self.gpumd_dir)
        #利用不确定性和能量差距筛选



        max_run_steps = self.idata.get("max_run_steps", 1200000)
        #切换系综，时间步骤重新设置
        #稀有事件，大更新
        max_iter = self.idata.get("max_iter", 20)
        if self.run_steps > max_run_steps:
            if self.ii >= max_iter:
                dlog.info(f"reached max iteration:{max_iter}, finished")
                exit()
            else:
                self.run_steps = max_run_steps / self.run_steps_factor

            # self.run_steps = self.idata.get("ini_run_steps", 100000)
            # self.model_devi_general_id += 1
            # with open(f"{self.work_dir}/model_devi_general_id.txt", "a") as f:
            #     np.savetxt(f, [self.model_devi_general_id], fmt="%d")
        self.write_steps()
        return
    
    def handle_bad_job(self,failed_row_indices=None,failed_task_dirs=None,allowed_shortest_distance=0.5, run_temp = 1500):
        """
        rerun the failed task
        """
        work_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd/mattersim"
        if not os.path.exists(work_dir):
            os.makedirs(work_dir,exist_ok=True)
        os.chdir(work_dir)
        dlog.info(f"the failed job will be rerun in {os.getcwd()}")
        task_dirs = []
        os.system(f"rm -rf task.*")
        for ii,failed_row_index in enumerate(failed_row_indices):
            if failed_row_index < 3:
                raise ValueError(f"the first 4 frames are failed, please check the input of {failed_task_dirs[ii]}")
            task_dir = os.path.join(work_dir, f"task.{ii:06d}")
            task_dirs.append(task_dir)
            os.makedirs(task_dir, exist_ok=True)
            os.chdir(task_dir)

            dlog.info(f"the {failed_row_index-2} frames of {failed_task_dirs[ii]} will be rerun")
            atoms = read(os.path.join(failed_task_dirs[ii], "dump.xyz"), index = failed_row_index-2)
            struc_file = os.path.abspath(os.path.join(task_dir, "POSCAR"))
            write(struc_file, atoms)
            py_file = continue_pytemplate.format(structure = struc_file,temperature = run_temp, steps = 20000)
            with open(os.path.join(task_dir, "ensemble.py"), "w",encoding="utf-8") as f:
                f.write(py_file)

        os.chdir(work_dir)
        task_dirs = [os.path.abspath(task_dir) for task_dir in glob("task.*")]
        
        task_dirs.sort()

        sorted_task_dirs = copy.deepcopy(task_dirs)

        for task_dir in task_dirs:
            os.chdir(task_dir)
            if os.path.exists("task_finished"):
                sorted_task_dirs.remove(task_dir)
                dlog.info(f"the task {task_dir} has been finished")

        os.chdir(work_dir)

        self.run_pytasks(sorted_task_dirs)
        
        os.chdir(work_dir)
        trajs = []
        for task_dir in task_dirs:
            traj_file = os.path.join(task_dir, "out.traj")
            traj = read(traj_file, index = ":")
            trajs.extend(traj)
            
        self.max_candidate = self.idata.get("max_candidate", 1000)
        
        if len(trajs) > self.max_candidate:
            # 随机抽取max_candidate个样本
            random_indices = random.sample(range(len(trajs)), self.max_candidate)
            random_indices.sort()  # 可选，保持帧的顺序
            trajs = [trajs[i] for i in random_indices]
        shortest_distances = [] 
        for ii, atoms in enumerate(trajs):
            shortest_distance = get_shortest_distance(atoms)
            shortest_distances.append(shortest_distance)
        np.savetxt("shortest_distance.txt", shortest_distances, fmt = "%12.2f")
        if np.any(np.array(shortest_distances) < allowed_shortest_distance):
            dlog.error(f"some frames have shortest distance less than {allowed_shortest_distance}")
            raise ValueError(f"some frames have shortest distance less than {allowed_shortest_distance}")

        label_dir = os.path.join(self.iter_dir, "02.label")
        os.makedirs(label_dir, exist_ok=True)
        os.chdir(label_dir)
        with open("candidate.xyz", "a") as f:
            write_extxyz(f, trajs)

    def run_pytasks(self, task_dirs):
        self.gpu_available = self.idata.get("gpu_available", [0, 1, 2, 3])
        self.gpu_per_task = self.idata.get("gpu_per_task", 1)
        python_interpreter = self.idata.get("python_interpreter")
        processes = []
        
        # 限制同时运行的任务数量
        max_concurrent_tasks = len(self.gpu_available)  # 或者其他合理的数量
        
        for i in range(0, len(task_dirs), max_concurrent_tasks):
            batch_dirs = task_dirs[i:i + max_concurrent_tasks]
            batch_processes = []
            
            for task_dir in batch_dirs:
                os.chdir(task_dir)
                if os.path.exists("task_finished"):
                    dlog.warning(f"{task_dir} has already been finished, skip it")
                    continue
                    
                basename = "ensemble.py"
                gpu_id = self.gpu_available[len(batch_processes) % len(self.gpu_available)]
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
                log_file = os.path.join(task_dir, 'log')
                try:
                    with open(log_file, 'w') as log:
                        process = subprocess.Popen(
                            [python_interpreter, basename],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=env
                        )
                        batch_processes.append((process, task_dir))
                except Exception as e:
                    dlog.error(f"Failed to start process in {task_dir}: {str(e)}")
                    raise Exception(f"Failed to start process in {task_dir}: {str(e)}")

            
            # 添加超时机制和健康检查
            timeout = 3600  # 设置合理的超时时间（秒）
            start_time = time.time()
            
            while batch_processes and time.time() - start_time < timeout:
                for i, (process, task_dir) in enumerate(batch_processes[:]):
                    ret = process.poll()
                    if ret is not None:  # 进程已结束
                        if ret != 0:
                            dlog.error(f"Process failed with return code {ret}. Check log at: {task_dir}/log")
                        else:
                            try:
                                os.chdir(task_dir)
                                ase_plt()
                                os.system(f"touch {task_dir}/task_finished")
                                dlog.info(f"Process completed successfully. Log at: {task_dir}/log")
                            except Exception as e:
                                dlog.error(f"Post-processing failed for {task_dir}: {str(e)}")
                                raise Exception(f"Post-processing failed for {task_dir}: {str(e)}")
                        
                        batch_processes.pop(i)
                
                # 防止 CPU 空转
                if batch_processes:
                    time.sleep(1)
            
            # 处理超时的进程
            for process, task_dir in batch_processes:
                if process.poll() is None:  # 进程仍在运行
                    dlog.warning(f"Process in {task_dir} timed out, terminating...")
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        dlog.error(f"Process in {task_dir} still running after termination, killing...")
                        process.kill()  # 强制终止
                    dlog.error(f"Process in {task_dir} terminated due to timeout")

    # def run_pytasks(self,task_dirs):
    #     self.gpu_available = self.idata.get("gpu_available",[0,1,2,3])
    #     self.gpu_per_task = self.idata.get("gpu_per_task", 1)
    #     python_interpreter = self.idata.get("python_interpreter")
    #     processes = []
        
    #     for index,task_dir in enumerate(task_dirs):
    #         os.chdir(task_dir)
    #         if os.path.exists("task_finished"):
    #             dlog.warning(f"{task_dir} has already been finished, skip it")
    #             continue
    #         # basename = os.path.basename(file)
    #         basename = "ensemble.py"
    #         gpu_id = self.gpu_available[index%len(self.gpu_available)]
    #         env = os.environ.copy()
    #         env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    #         # Open subprocess and redirect stdout and stderr to a log file
    #         log_file = os.path.join(task_dir, 'log')  # Log file path
    #         with open(log_file, 'w') as log:
    #             process = subprocess.Popen(
    #                 [python_interpreter, basename], 
    #                 stdout=log, 
    #                 stderr=subprocess.STDOUT,  # Combine stderr with stdout
    #                 env = env
    #             )
    #             processes.append((process, task_dir))  # Store the process and log file

    #     # Wait for all subprocesses to complete and check for errors
    #     for process, task_dir in processes:
    #         process.wait()  # Wait for the process to complete
    #         # Check for errors using the return code
    #         if process.returncode != 0:
    #             dlog.error(f"Process failed. Check the log at: {task_dir}/log")
    #             raise RuntimeError(f"Process failed. Check the log at: {task_dir}/log")
    #         else:
    #             os.chdir(task_dir)
    #             ase_plt()
    #             os.system(f"touch {task_dir}/task_finished")
    #             dlog.info(f"Process completed successfully. Log saved at: {task_dir}/log")
        
    @classmethod
    def relative_force_error(cls, total_time, nep_dir = None, mode:str = "mean", level = 1 ,allowed_shortest_distance = 0.55, allowed_max_temp = 7000):
        """
        return frame_index dict for accurate, candidate, failed
        the candidate_index may be more than the needed, need to resample
        """

        if os.path.exists("frame_property.txt"):
            frame_property = np.loadtxt("frame_property.txt")

            if os.path.exists("failed_index.txt"):
                with open("failed_index.txt","r") as f:
                    failed_index = int(f.read())
            else :
                failed_index = len(frame_property)

            atoms_list = read(f"dump.xyz", index = ":")
            return  atoms_list, frame_property, failed_index

        if not nep_dir:
            nep_dir = os.getcwd()
        calculator_fs = glob(f"{nep_dir}/**/nep*.txt")
        atoms_list = read(f"dump.xyz", index = ":")
        f_lists = force_main(atoms_list, calculator_fs)
        property_list = []
        time_list = np.linspace(0, total_time, len(atoms_list), endpoint=False)/1000
        for ii,atoms in tqdm(enumerate(atoms_list)):
            f_list = [item[ii] for item in f_lists]
            energy_list = [iterm[1] for iterm in f_list]
            f_list = [iterm[0] for iterm in f_list]
            f_avg = np.average(f_list,axis=0)
            df_sqr = np.sqrt(np.mean(np.square([(f_list[i] - f_avg) for i in range(len(f_list))]).sum(axis=2),axis=0))
            abs_f_avg = np.mean(np.sqrt(np.square(f_list).sum(axis=2)),axis=0)
            relative_error = (df_sqr/(abs_f_avg+level))
            if mode == "mean":
                relative_error = np.mean(relative_error)
            elif mode == "max":
                relative_error = np.max(relative_error)
            energy_error = np.sqrt(np.mean(np.power(np.array(energy_list)-np.mean(energy_list),2)))
            shortest_distance = get_shortest_distance(atoms)
            property_list.append([time_list[ii], relative_error, energy_error, shortest_distance])
        # property_list
        property_list_np = np.array(property_list)
        thermo = np.loadtxt("thermo.out")
        thermo_new = compute_volume_from_thermo(thermo)[:,[0,2,3,-1]]
        frame_property = np.concatenate((property_list_np,thermo_new),axis=1)
        temperatures = thermo[:,0]
        shortest_distances = frame_property[:,3]
        result = np.where(np.logical_or(temperatures > allowed_max_temp, shortest_distances < allowed_shortest_distance))
        if result[0].size > 0:
            # 获取第一个大于6000的数的行索引
            first_row_index = result[0][0]
        else:
            first_row_index = len(frame_property)  # 如果没有找到任何大于6000的数

        fmt = "%12.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f"
        header = f"{'time':^9} {'relative_error':^14} {'energy_error':^14} {'shortest_d':^14} {'temperature':^14} {'potential':^14} {'pressure':^14} {'volume':^14}"
        np.savetxt("frame_property.txt", frame_property, fmt = fmt, header = header)
        
        if first_row_index != len(frame_property):
            new_total_time = time_list[first_row_index-1]*1000
            dlog.warning(f"thermo.out has temperature > {allowed_max_temp} K or shortest distance < {allowed_shortest_distance} too early at frame {first_row_index}({new_total_time} ps), the gpumd task should be rerun")
            np.savetxt(f"failed_index.txt", [first_row_index], fmt="%12d")
            return atoms_list, frame_property, first_row_index
            
        return atoms_list, frame_property, first_row_index

    def make_label_task(self):
        label_engine = self.idata.get("label_engine","mattersim")
        if label_engine == "mattersim":
            self.make_mattersim_task()
        elif label_engine == "vasp":
            self.make_vasp_task()
        else:
            raise NotImplementedError(f"The label engine {label_engine} is not implemented")        

    def make_mattersim_task(self):
        """
        change the calculator to mattersim, and write the train.xyz
        """

        # 将工作目录设置为iter_dir下的02.label文件夹
        iter_dir = self.iter_dir
        work_dir = os.path.join(iter_dir, "02.label")
        train_ratio = self.idata.get("training_ratio", 0.8)
        os.chdir(work_dir)
        atoms_list = read("candidate.xyz", index=":", format="extxyz")
        self.pot_file = self.idata.get("pot_file")
        # 创建MatterSimCalculator对象，用于计算原子能量
        Nepactive.run_mattersim(atoms_list=atoms_list, pot_file=self.pot_file, train_ratio=train_ratio)

    @classmethod
    def run_mattersim(cls, atoms_list:List[Atoms], pot_file:str, train_ratio:float=0.8,tqdm_use:Optional[bool]=False):
        calculator = MatterSimCalculator(load_path=pot_file,device="cuda")
        if os.path.exists("candidate.traj"):
            os.remove("candidate.traj")
        traj = Trajectory('candidate.traj', mode='a')

        def change_calc(atoms:Atoms):
            atoms._calc=calculator
            atoms.get_potential_energy()
            traj.write(atoms)
            return atoms
        # 对每个原子调用change_calc函数，将计算结果写入Trajectory对象
        # atoms = [change_calc(atoms[i]) for i in tqdm(range(len(atoms)))]
        if tqdm_use:
            atoms = [change_calc(atoms_list[i]) for i in tqdm(range(len(atoms_list)))]
        else:
            atoms = [change_calc(atoms_list[i]) for i in range(len(atoms_list))]   
        # 读取Trajectory对象中的原子信息
        atoms = read("candidate.traj",index=":")
        train:List[Atoms]=[]
        test:List[Atoms]=[]
        failed:List[Atoms]=[]
        failed_index=[]
        for i in range(len(atoms)):
            rand=random.random()
            if np.max(np.abs(atoms[i].get_forces())) > 60:
                failed.append(atoms[i])
                failed_index.append(i)
            elif rand <= train_ratio:
                train.append(atoms[i])
            elif rand > train_ratio:
                test.append(atoms[i])
            else:
                dlog.warning(f"{atoms[i]}failed to be classified")
        if train:
            write_extxyz("iter_train.xyz",train)
        if test:
            write_extxyz("iter_test.xyz",test)
        if failed:
            write_extxyz("iter_failed.xyz",failed)
            np.savetxt("failed_index.txt",failed_index,fmt="%12d")
        # 将原子信息写入train_iter.xyz文件
        dlog.warning(f"failed structures:{len(failed_index)}")
        del calculator
        empty_cache()

    def make_vasp_task(self):
        task = Remotetask(iternum = self.ii, idata = self.idata, trajfile = self.trajfile)
        task.run_submission(jj=3)

    def run_label_task(self):
        label_engine = self.idata.get("label_engine","mattersim")
        if label_engine == "mattersim":
            return
        elif label_engine == "vasp":
            self.run_vasp_task()

    def run_vasp_task(self):
        assert os.path.isabs(self.iter_dir)
        os.chdir(f"{self.iter_dir}/02.label")
        if task is None:
            task = Remotetask(idata = self.idata)
        task.run_submission(jj=4)

    def write_steps(self):
        with open(f"{self.work_dir}/steps.txt","a") as f:
            f.write(f"{int(self.run_steps):12d}\n")
        np.savetxt(f"{self.work_dir}/iter.{self.ii:06d}/steps.txt",np.array([self.run_steps]),fmt="%12d")