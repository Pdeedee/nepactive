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
from nepactive.template import npt_template,nphugo_template,nvt_template,msst_template,model_devi_template
from nepactive.gpumd_plt import gpumdplt
from nepactive import parse_yaml

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

def get_force(atoms:Atoms,calculator):
    atoms._calc=calculator
    return atoms.get_forces(),atoms.get_potential_energy()

def compute_volume_from_thermo(thermo:np.ndarray):
    num_columns = thermo.shape[1]
    # dlog.info(f"thermo file has {num_columns} columns")
    volume:np.ndarray
    if num_columns == 12:
        box_length_x = thermo[:, 9]
        box_length_y = thermo[:, 10]
        box_length_z = thermo[:, 11]
        volume = np.abs(box_length_x * box_length_y * box_length_z)
        thermo = np.insert(thermo, 12, volume, axis=1)[:,[0,1,2,3,12]]
    elif num_columns == 18:
        ax, ay, az = thermo[:, 9], thermo[:, 10], thermo[:, 11]
        bx, by, bz = thermo[:, 12], thermo[:, 13], thermo[:, 14]
        cx, cy, cz = thermo[:, 15], thermo[:, 16], thermo[:, 17]
        # 计算晶胞的体积（使用行列式公式）
        # 叉积 (b x c)
        bx_cy_bz = by * cz - bz * cy
        bx_cz_by = bz * cx - bx * cz
        bx_cx_by = bx * cy - by * cx
        # 点积 a · (b x c)
        volume = ax * bx_cy_bz + ay * bx_cz_by + az * bx_cx_by
        volume = np.abs(volume)  # 体积取绝对值
        thermo = np.insert(thermo, 18, volume, axis=1)[:,[0,1,2,3,18]]
    else:
        raise ValueError("thermo file has wrong format")
    return thermo

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
        # Change the working directory to the init directory
        work_dir = f"{self.work_dir}/init"
        ase_ensemble_files:list[str] = [os.path.abspath(path) for path in self.idata.get("ini_ase_ensemble_files")]
        assert ase_ensemble_files
        # dlog.info(f"ase_ensemble_files:{ase_ensemble_files}")
        # print(ase_ensemble_files)
        python_interpreter:str = self.idata.get("python_interpreter")
        processes = []
        self.pot_file:str = self.idata.get("pot_file")
        pot_file =self.pot_file
        self.gpu_available = self.idata.get("gpu_available")
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        # Make initial training task directories
        for index, file in enumerate(ase_ensemble_files):
            
            task_name = f"task.{index:06d}"
            
            # Ensure the task directory is created
            task_dir = os.path.join(work_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)  # Create the task directory, if it doesn't exist
            
            # shutil.copy(file, task_dir)
            # os.symlink(pot_file, task_dir)
            os.system(f"ln -snf {pot_file} {task_dir}/model.pth")
            os.system(f"ln -snf {file} {task_dir}")
            os.chdir(task_dir)
        
            # Get the basename of the file (e.g., "npt.py")
            basename = os.path.basename(file)
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
                processes.append((process, log_file))  # Store the process and log file

        # Wait for all subprocesses to complete and check for errors
        for process, log_file in processes:
            process.wait()  # Wait for the process to complete
            # Check for errors using the return code
            if process.returncode != 0:
                dlog.error(f"Process failed. Check the log at: {log_file}")
                raise RuntimeError(f"Process failed. Check the log at: {log_file}")
            else:
                dlog.info(f"Process completed successfully. Log saved at: {log_file}")

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
        fs = glob("init/task.*/*.traj")
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
        # for ii in range(iter_numbers):
        #     iter_dir = os.path.abspath(os.path.join(work_dir,f"iter.{ii:06d}")) #the work path has been changed
        #     os.makedirs(iter_dir, exist_ok=True)
        #     make_nep_train(ii=ii, iter_dir=iter_dir, pot_num=pot_num, nep_template=nep_template, init=init, idata = idata)
        #     gpumd_dir = make_gpumd_run(iter_idr=iter_dir, pot_num=pot_num, idata = idata)
        #     post_gpumd_run(gpumd_dir=gpumd_dir, iter_dir=iter_dir, threshold_low=threshold_low, threshold_high=threshold_high, idata = idata)
        #     maker_label_run(iter_dir=iter_dir, idata = idata)

        record = "record.nep"
        iter_rec = [0, -1]
        if os.path.isfile(record):
            with open(record) as frec:
                for line in frec:
                    iter_rec = [int(x) for x in line.split()]
        cont = True
        self.ii = -1
        numb_task = 6
        max_tasks = 10000
        while cont:
            self.ii += 1
            if self.ii < iter_rec[0]:
                continue
            self.iter_dir = os.path.abspath(os.path.join(self.work_dir,f"iter.{self.ii:06d}")) #the work path has been changed
            # iter_num = self.ii
            os.makedirs(self.iter_dir, exist_ok=True)
            iter_name = f"iter.{self.ii:06d}"
            # iter_name = make_iter_name(ii)
            # sepline(iter_name, "=")
            for jj in range(numb_task):
                self.jj = jj
                yaml_synchro = self.idata.get("yaml_synchro", False)
                if yaml_synchro:
                    dlog.info(f"yaml_synchro is True, reread the in.yaml from {self.work_dir}/in.yaml")
                    self.idata:dict = parse_yaml(f"{self.work_dir}/in.yaml")
                if self.ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1]:
                    continue
                task_name = "task %02d" % jj
                sepline(f"{iter_name} {task_name}", "-")
                if jj == 0:
                    # log_iter("make_train", ii, jj)
                    self.make_nep_train()
                elif jj == 1:
                    cont = self.make_model_devi()
                elif jj == 2:
                    dlog.info(f"start the {self.ii}th gpumd task")
                    self.run_gpumd_task()
                    dlog.info(f"finish the {self.ii}th gpumd task")
                elif jj == 3:
                    self.post_gpumd_run()
                elif jj == 4:
                    self.make_label_task()
                elif jj == 5:
                    self.run_label_task()
                else:
                    raise RuntimeError("unknown task %d, something wrong" % jj)
                
                os.chdir(self.work_dir)
                record_iter(record, self.ii, jj)

    def make_nep_train(self):
        '''
        Train nep. 
        '''
        #ensure the work_dir is the absolute pathk
        global_work_dir = os.path.abspath(self.work_dir)
        work_dir = os.path.abspath(os.path.join(self.iter_dir, "00.nep"))
        pot_num = self.idata.get("pot_num", 4)
        init:bool = self.idata.get("init", True)
        pot_inherit:bool = self.idata.get("pot_inherit", True)
        nep_template = os.path.abspath(self.idata.get("nep_template"))
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        #     absworkdir/iter.000000/00.nep/
        processes = []

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
            if not files:
                raise ValueError("No files found to merge.")
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
                os.symlink(nep_template, "nep.in")
            if pot_inherit and self.ii > 0:
                nep_restart = f"{self.work_dir}/iter.{self.ii-1:06d}/00.nep/task.{jj:06d}/nep.restart"
                dlog.info(f"pot_inherit is true, will copy nep.restart from {nep_restart}")
                shutil.copy(nep_restart, "nep.restart")
                # exit()
            log_file = os.path.join(task_dir, 'subprocess_log.txt')  # Log file path
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

        # work_dir = os.path.abspath(f"{work_dir}/task.{ii:06d}")

    def make_model_devi(self):
        '''
        run gpumd, this function is referenced by make_loop_train
        '''
        # model_devi_jobs = idata["model_devi"]
        model_devi_general:list[dict] = self.idata.get("model_devi_general", None)
        # if self.make_gpumd_task_first:
        #     dlog.info(f"make_gpumd_task_first is true, will remove old task files {self.work_dir}/iter.{iter_index:06d}/01.gpumd/task*")
        #     os.system(f"rm -r {self.work_dir}/iter.{iter_index:06d}/01.gpumd/task*")
        dlog.info(f"model_devi_general:{model_devi_general}")

        model_devi = self.get_model_devi()


        iter_index = self.ii
        # os.chdir(f"")
        if self.make_gpumd_task_first:
            dlog.info(f"make_gpumd_task_first is true, will remove old task files {self.work_dir}/iter.{iter_index:06d}/01.gpumd/task*")
            os.system(f"rm -r {self.work_dir}/iter.{iter_index:06d}/01.gpumd/task*")
        # dlog.info(f"model_devi:{model_devi}")
        work_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
        self.make_gpumd_task(model_devi=model_devi,work_dir=work_dir)


    def make_gpumd_task(self,model_devi:dict,work_dir:str=None):
        if not work_dir:
            work_dir = os.getcwd()
        if not model_devi:
            model_devi = self.idata
        all_dict = {}
        ensemble = model_devi.get("ensemble")[0]
        structure_id = model_devi.get("structure_id")
        structure_files:list = self.idata.get("structure_files")
        structure_prefix = self.idata.get("structure_prefix")
        structure_files = [os.path.join(structure_prefix,structure_file) for structure_file in structure_files]
        assert structure_files is not None
        # dump_freq = model_devi.get("dump_freq")
        # if not dump_freq:
        #     dump_freq = self.idata.get("dump_freq")
        structure = [structure_files[ii] for ii in structure_id] #####################################################
        all_dict["structure"] = structure
        time_step_general = self.idata.get("model_devi_time_step", None)
        time_step = model_devi.get("time_step")
        run_steps = model_devi.get("run_steps",20000)
        if not time_step:
            time_step = time_step_general
        assert all([structure,time_step,run_steps]), "有变量为空"
        # task_dict = {}
        task_dicts = []
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
                e0 = model_devi.get("e0")
                p0 = model_devi.get("p0")
                v0 = model_devi.get("v0")
                all_dict["e0"] = e0
                all_dict["p0"] = p0
                all_dict["v0"] = v0
        elif ensemble == "msst":
            v_shock:list = model_devi.get("v_shock",[9.0])       #
            qmass:list = model_devi.get("qmass",[100000])           #
            viscosity:list = model_devi.get("viscosity",[10])   #
            all_dict["v_shock"] = v_shock
            all_dict["qmass"] = qmass
            all_dict["viscosity"] = viscosity
        else:
            raise NotImplementedError
        dlog.info(f"all_dict:{all_dict}")
        assert all(v not in [None, '', [], {}, set()] for v in all_dict.values())
        # task_para = []
        # combo_numbers = 0
        for combo in itertools.product(*all_dict.values()):
            # 将每一组合生成字典，字典的键是列表的变量名，值是组合中的对应元素
            combo_dict = {keys: combo[index] for index, keys in enumerate(all_dict.keys())}
            task_dicts.append(combo_dict)
        dlog.info(f"{all_dict.keys()} generate {len(task_dicts)} tasks")
        self.needed_frames = self.idata.get("needed_frames",10000)
        self.frames_pertask = self.needed_frames/len(task_dicts)
        self.dump_freq = max(1,floor(run_steps/self.frames_pertask))
        dump_freq = self.dump_freq
        self.model_devi_task_numbers = len(task_dicts)
        nep_file = self.idata.get("nep_file","../../00.nep/task.000000/nep.txt")
        for index,task in enumerate(task_dicts):
            if ensemble == "msst":
                text = msst_template.format(time_step = time_step,run_steps = run_steps,dump_freq = dump_freq, **task)
            elif ensemble == "nvt":
                text = nvt_template.format(time_step = time_step,run_steps = run_steps,dump_freq = dump_freq, **task)
            elif ensemble == "npt":
                text = npt_template.format(time_step = time_step,run_steps = run_steps,dump_freq = dump_freq, **task)
            elif ensemble == "nphugo":
                assert run_steps > 20000
                text = nphugo_template.format(
                    time_step = time_step,
                    run_steps = run_steps-20000,
                    dump_freq = dump_freq,
                    **task
                )
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
                os.symlink(structure,f"model.xyz")
            with open(file,mode='w') as f:
                f.write(text)
            if not os.path.isfile("nep.txt"):
                os.symlink(nep_file, "nep.txt")
                
        # dlog.info("generate gpumd task done")

    def run_gpumd_task(self):
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
        tasks = glob("task.*")
        # tasks = glob("task.*")
        if not tasks:
            raise RuntimeError(f"No task files found in {model_devi_dir}")
        
        gpu_available = self.idata.get("gpu_available")
        self.task_per_gpu = self.idata.get("task_per_gpu")
        parallel_process = self.task_per_gpu * len(gpu_available)
        interval = ceil(len(tasks)/parallel_process)
        jobs = [tasks[i:i + interval] for i in range(0, len(tasks), interval) if tasks[i:i + interval]]
        job_list = []
        for index,job in enumerate(jobs):
            # text = ""
            # for task in job:
            #     text += model_devi_template.format(work_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd",
            #                                       task_dir = task)
            text = "".join([model_devi_template.format(work_dir=f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd", task_dir=task) for task in job])
            with open(f"job_{index:03d}.sub", 'w') as f:
                f.write(text)
            job_list.append(f"job_{index:03d}.sub")
        processes = []
        dlog.info(f"divide {len(tasks)} tasks into {len(job_list)} jobs")
        # model_devi_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
        for index,job in enumerate(job_list):
            # os.chdir(f"{model_devi_dir}/{task}")    
            log_file = f"job_{index:03d}.log"  # Log file path
            env = os.environ.copy()
            gpu_id = gpu_available[ceil(index/self.task_per_gpu)]
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    ["bash", job],  # 程序名
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env  # 使用修改后的环境
                )
                processes.append((process, log_file))
                dlog.info(f"submitted {job} to GPU:{gpu_id}")

        for process, log_file in processes:
            process.wait()  # Wait for all processes to complete
            # Check for errors using the return code
            if process.returncode != 0:
                dlog.error(f"Process failed. Check the log at: {log_file}")
                raise RuntimeError(f"One or more processes failed. Check the log ({log_file}) files for details.")
            else: 
                dlog.info(f"gpumd run successfully. Log saved at: {log_file}")

                plot = self.idata.get("gpumd_plt",True)

        if plot:
            model_devi_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
            os.chdir(model_devi_dir)
            task_dirs = [os.path.join(model_devi_dir,task) for task in glob("task*")]
            dlog.info(f"plotting {len(task_dirs)} tasks")
            for task_dir in task_dirs:
                #ensure the work_dir is the absolute path
                os.chdir(task_dir)
                self.time_step = self.idata.get("time_step")
                gpumdplt(self.dump_freq,self.time_step)
        
    def get_model_devi(self):
        '''
        get the model deviation from the gpumd run
        '''
        model_devi_general = self.idata.get("model_devi_general", None)
        run_steps_length = 0
        for ii, model_devi in enumerate(model_devi_general):
            model_devi_id = process_id(model_devi.get("id", None))
            each_run_steps = model_devi.get("each_run_steps", [0])
            now_run_steps_length = len(model_devi_id)
            if self.ii in model_devi_id:
                find_id = True
                model_devi = model_devi_general[ii]
                runsteps_id =  self.ii - run_steps_length
                # dlog.info(f"runsteps_id:{runsteps_id},self.ii:{self.ii},run_steps_length:{run_steps_length}")
                if runsteps_id < 0:
                    runsteps_id = -1
                    dlog.info("the each_runsteps is not enough, will use the last one")
                model_devi["run_steps"] = each_run_steps[runsteps_id]
                dlog.info(f"{ii}th model_devi_general has run_steps:{model_devi['run_steps']}")
                break
            run_steps_length += now_run_steps_length   #skip the run_steps_length
            
        if not find_id:
            dlog.info(f"the ii:{self.ii} is not in model_devi_general, will use the last one")
            dlog.warning(f"not finding the {self.ii}th task settings in model_devi_general, will end the job")
            raise RuntimeError(f"not finding the {self.ii}th task settings in model_devi_general, will end the job")
        return model_devi

    

    def post_gpumd_run(self):
        '''
        extract the information from gpumd run, this function is referenced by make_loop_train
        '''
        try:
            self.dump_freq
        except AttributeError:
            # 捕获变量不存在时的错误
            self.make_gpumd_task_first = False
            self.make_model_devi()
            self.make_gpumd_task_first = True
            dlog.info("remake the gpumd task")
        iter_dir = self.iter_dir
        gpumd_dir = os.path.join(self.iter_dir, "01.gpumd")
        os.chdir(gpumd_dir)
        subprocess.run(["cat */dump.xyz > all.xyz"], shell = True, capture_output = True, text=True)
        # subprocess.run(["cat */thermo.out > thermo.out"], shell = True, capture_output = True, text=True)
        dlog.info("-----start analysis the trajectory-----")
        atoms = read("all.xyz", index=":", format="extxyz")
        dlog.info(f"Totally {len(atoms)} structures.")
        model_devi:dict = self.get_model_devi()
        max_candidate = model_devi.get("max_candidate")
        continue_from_old = self.idata.get("continue_from_old", False)
        if not max_candidate:
            max_candidate = self.idata.get("max_candidate")
        #检查效率，是重新复制粘贴快，不爆炸显存的话应该复制粘贴快吧
        if len(atoms) > max_candidate*2:
            random_indices = random.sample(range(len(atoms)),max_candidate*2)
        else:
            random_indices = range(len(atoms))

        # if len(atoms) > 200:
        #     random_indices = random.sample(range(len(atoms)),200)
        # else:
        #     random_indices = range(len(atoms))
        dlog.info(f"extract {len(random_indices)} structures.")

        assert(os.path.isabs(iter_dir))

        nep_dir = os.path.join(self.iter_dir, "00.nep")
        level = self.idata.get("uncertainty_level", 1)
        threshold = self.idata.get("uncertainty_threshold", [0.2,0.5])
        mode = self.idata.get("uncertainty_mode", "mean")
        energy_threshold = self.idata.get("energy_threshold", None)
        uncertainty, frame_index= Nepactive.relative_force_error( atoms_list=atoms, nep_dir = nep_dir, selected_indices=random_indices, level = level, threshold = threshold, mode = mode, energy_threshold = energy_threshold)
        # dlog.info(f"uncertainty:{uncertainty},frame_index:{frame_index}")

        #剔除多余的candidate
        original_failed_ratio = len(frame_index["failed"])/(len(random_indices))
        if len(frame_index["candidate"]) > max_candidate:
            candidate_indices = random.sample(frame_index["candidate"], max_candidate)
            frame_index["candidate"] = candidate_indices
            frame_index["failed"].extend([ii for ii in frame_index["candidate"] if ii not in candidate_indices])
        accurate_ratio = len(frame_index["accurate"])/(len(random_indices))
        dlog.info(f"accurate:{len(frame_index['accurate'])}, candidate:{len(frame_index['candidate'])}, failed:{len(frame_index['failed'])}, accurate_ratio:{accurate_ratio},original_failed:{original_failed_ratio}")


        candidate_info = frame_index["candidate"]
        uncertainty = [iterm[1] for iterm in candidate_info]
        with open("frame_index.json", "w") as json_file:
            json.dump(frame_index, json_file, indent=4)

        index = [iterm[0] for iterm in candidate_info]
        traj_selected = [atoms[i] for i in index]
        shortest_distance = [get_shortest_distance(atom) for atom in traj_selected]

        thermo_files = glob("*/thermo.out")
        thermo_files.sort()
        thermo_averages = []
        for iter,thermo_file in enumerate(thermo_files):
            dlog.info(f"processing thermo_file:{thermo_file}")
            if iter == 0:
                thermo = np.loadtxt(thermo_file)
                thermo = compute_volume_from_thermo(thermo)
                thermos = thermo
                thermo_averages = thermo_average = np.average(thermo[int(0.2 * len(thermo)):int(0.9 * len(thermo))], axis=0)[np.newaxis, :]
            else:
                thermo = np.loadtxt(thermo_file)
                thermo = compute_volume_from_thermo(thermo)
                thermos = np.concatenate((thermos,thermo),axis=0)
                thermo_average = np.average(thermo[int(0.2 * len(thermo)):int(0.9 * len(thermo))], axis=0)[np.newaxis, :]
                thermo_averages = np.concatenate((thermo_averages,thermo_average),axis=0)

        property_needed = thermos[:,[0,3,4]]
        property_needed = property_needed[index]
        self.frames_pertask = len(thermos)/self.model_devi_task_numbers
        property_needed = np.insert(property_needed, 0, index, axis=1)
        property_needed = np.insert(property_needed, property_needed.shape[1], shortest_distance, axis=1)
        property_needed = np.insert(property_needed, property_needed.shape[1], uncertainty, axis=1)
        sorted_indices = np.argsort(property_needed[:, 0])
        # 根据索引重新排序整个数组
        property_needed = property_needed[sorted_indices]
        property_needed[:,0] = property_needed[:,0]%self.frames_pertask
        np.savetxt("candidate.txt",property_needed,fmt = "%12d %12.2f %12.2f %12.2f %12.2f %12.2f", header = "%11s %-14s %-14s %-14s %-14s %-14s"%("index", "temperature", "pressure", "volume", "shortest_distance", "uncertainty"))
        with open(f'{self.work_dir}/thermo.txt', 'a') as f:
            np.savetxt(f, thermo_averages, fmt='%24.2f %12.2f %12.2f %12.2f %12.2f', 
                    header=" %14s %-14s %-14s %-14s %-14s" % ("temperature", "kinetic", "potential", "pressure", "volume"),
                    comments=f"#iter.{self.ii:06d}")
        os.remove("all.xyz")
        index.sort()
        atoms = [atoms[i] for i in index]
        label_dir = os.path.join(iter_dir, "02.label")
        os.makedirs(label_dir, exist_ok=True)
        write_extxyz(os.path.join(label_dir, "candidate.xyz"), atoms)
        if len(frame_index["candidate"]) == 0:
            # return True
            dlog.warning("no candidate frames is selected")
            
            raise ValueError("there are no candidate frames, maybe you can change the ensemble file")
        dlog.info("-----finish analysis the trajectory-----")

        del atoms
        empty_cache()

    @classmethod
    def relative_force_error(cls, atoms_list:List[Atoms], nep_dir = None, selected_indices=None, level = 1, threshold = [0.2, 0.5], mode:str = "mean" , energy_threshold = 1):
        """
        return uncertanty list for selected_indices in atoms traj
        return frame_index dict for accurate, candidate, failed
        the candidate_index may be more than the needed, need to resample
        """
        threshold_low = threshold[0]
        threshold_high = threshold[1]
        if not selected_indices:
            selected_indices = range(len(atoms_list))
        if not nep_dir:
            nep_dir = os.getcwd()
        calculator_fs = glob(f"{nep_dir}/**/nep*.txt")
        calculators = [NEP(calculator_fs[i]) for i in range(len(calculator_fs))]

        frame_index={
            "accurate":[],
            "candidate":[],
            "failed":[]
        }
        
        candidate_but_failed_index = []
        candidate_but_failed_atom_index = []
        natoms = len(atoms_list[0])
        
        for ii in selected_indices:
        # for ii in tqdm(random_indices):
            forceslist = [get_force(atoms_list[ii],calculator) for calculator in calculators]
            energy_list = [iterm[1] for iterm in forceslist]
            forceslist = [iterm[0] for iterm in forceslist]
            # exit()
            avg_force = np.average(forceslist,axis=0)
            deltaforcesquare = np.sqrt(np.mean(np.square([(forceslist[i] - avg_force) for i in range(len(forceslist))]).sum(axis=2),axis=0))
            # 先计算绝对值再mean
            absforce_avg = np.mean(np.sqrt(np.square(forceslist).sum(axis=2)),axis=0)
            relative_error = (deltaforcesquare/(absforce_avg+level))
            energy_error = np.sqrt(np.mean(np.power(np.array(energy_list)-np.mean(energy_list),2)))
            # relative_energy_error = energy_error/np.mean(energy_list)
            
            if mode == "max":
                uncertainty = np.max(relative_error)
            elif mode == "mean":
                uncertainty = np.mean(relative_error)
            else:
                raise KeyError("mode must be 'max' or 'mean'")
            corr_deltaforcesquare = uncertainty * (absforce_avg[np.argmax(relative_error)]+level)
            max_force_error = np.max(deltaforcesquare)
            corr_absforce_avg = absforce_avg[np.argmax(relative_error)]
            frame_list = [ii, uncertainty, corr_absforce_avg, corr_deltaforcesquare, max_force_error, energy_error]
            if not energy_threshold:
                energy_ok = True
            elif energy_error < energy_threshold:
                energy_ok = True
            elif energy_error > energy_threshold:
                energy_ok = False
            else:
                raise KeyError("energy_threshold must be a number or None")

            if uncertainty < threshold_low and energy_ok:
                frame_index["accurate"].append(frame_list)
            elif uncertainty > threshold_high:
                frame_index["failed"].append(frame_list)
            else:
                shortest_distance = get_shortest_distance(atoms_list[ii],candidate_but_failed_atom_index)
                if shortest_distance < 0.6 or absforce_avg.any() > 75:
                    frame_index["failed"].append(frame_list)
                    candidate_but_failed_index.append(ii)
                else:
                    frame_index["candidate"].append(frame_list)
        # dlog.info(f"accurate_shape:{np.array(frame_index['accurate']).shape}, failed_shape:{np.array(frame_index['failed']).shape},candidate_shape:{np.array(frame_index['candidate']).shape}")
        
        fmt = "%12d %12.2f %12.2f %12.2f %12.2f %12.2f"
        header = "%12s %12s %12s %12s %12s %12s" % ("index", "uncertainty", "corr_absforce_avg", "corr_deltaforcesquare", "max_force_error", "energy_error")
        # 保存函数
        def save_sorted_data(filename, data, fmt, header):
            np.savetxt(filename, data, fmt=fmt, header=header)
        # 处理 "failed", "accurate", "candidate" 数据
        if len(frame_index["failed"]) > 0:
            failed_array = np.array(frame_index["failed"])
            sorted_indices = failed_array[:, 0].argsort()
            sorted_failed = failed_array[sorted_indices]
            save_sorted_data("failed_error.txt", sorted_failed, fmt, header)

        if len(frame_index["accurate"]) > 0:
            accurate_array = np.array(frame_index["accurate"])
            sorted_indices = accurate_array[:, 0].argsort()
            sorted_accurate = accurate_array[sorted_indices]
            save_sorted_data("accurate_error.txt", sorted_accurate, fmt, header)

        if len(frame_index["candidate"]) > 0:
            candidate_array = np.array(frame_index["candidate"])
            sorted_indices = candidate_array[:, 0].argsort()
            sorted_candidate = candidate_array[sorted_indices]
            save_sorted_data("candidate_error.txt", sorted_candidate, fmt, header)

        if candidate_but_failed_index:
            dlog.warning(f"candidate_but_failed_index:{candidate_but_failed_index}") #,candidate_but_failed_atom_index:{candidate_but_failed_atom_index[random_indices.index(candidate_but_failed_index[0])]}")
            atoms_candidate_but_failed = [atoms_list[i] for i in candidate_but_failed_index]
            write(f"candidate_but_failed.pdb",atoms_candidate_but_failed)
        del calculators
        del calculator_fs
        empty_cache()
        return uncertainty,frame_index

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
        for i in range(len(atoms)):
            rand=random.random()
            if rand <= train_ratio:
                train.append(atoms[i])
            elif rand > train_ratio:
                test.append(atoms[i])
            else:
                dlog.warning(f"{atoms[i]}failed to be classified")
        if train:
            write_extxyz("iter_train.xyz",train)
        if test:
            write_extxyz("iter_test.xyz",test)
        # 将原子信息写入train_iter.xyz文件
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
