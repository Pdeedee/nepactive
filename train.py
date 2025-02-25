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
from nepactive.template import npt_template,nphugo_template,nvt_template,msst_template,model_devi_template
from nepactive.gpumd_plt import gpumdplt

def get_shortest_distance(atoms:Atoms,atom_index=None):
    distance_matrix = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distance_matrix, np.inf)
    min_index = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
    if atom_index is not None:
        atom_index.append(min_index)
    return np.min(distance_matrix)

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

class Nepactive():
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
                if self.ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1]:
                    continue
                task_name = "task %02d" % jj
                sepline(f"{iter_name} {task_name}", "-")
                if jj == 0:
                    # log_iter("make_train", ii, jj)
                    self.make_nep_train()
                elif jj == 1:
                    # log_iter("run_train", ii, jj)
                    cont = self.make_gpumd_task()
                    # cont = make_model_devi(ii, jdata, mdata)
                    if not cont:
                        break
                elif jj == 2:
                    self.run_gpumd_task()
                elif jj == 3:
                    # log_iter("post_train", ii, jj)
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
                nep_in = f"{self.work_dir}/iter.{self.ii-1:06d}/00.nep/task.{jj:06d}/nep.txt"
                dlog.info(f"pot_inherit is true, will copy nep.txt from {nep_in}")
                shutil.copy(nep_in, "nep.txt")
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

    def make_gpumd_task(self):
        '''
        run gpumd, this function is referenced by make_loop_train
        '''
        # model_devi_jobs = idata["model_devi"]
        all_dict = {}
        model_devi:dict = self.idata["model_devi"][self.ii]
        iter_index = self.ii
        # os.chdir(f"")
        if self.make_gpumd_task_first:
            dlog.info(f"make_gpumd_task_first is true, will remove old task files {self.work_dir}/iter.{iter_index:06d}/01.gpumd/task*")
            os.system(f"rm -r {self.work_dir}/iter.{iter_index:06d}/01.gpumd/task*")
        # dlog.info(f"model_devi:{model_devi}")
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
            task_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd/task.{index:06d}"
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
                os.symlink(f"../../00.nep/task.000000/nep.txt", "nep.txt")
                
        # print(f"gpumd_templates:{gpumd_templates}")
        model_devi_jobs = self.idata["model_devi"]
        # gpumd_dir = os.path.join(self.iter_dir, "01.gpumd")
        if iter_index >= len(model_devi_jobs):
            cont = False
        else:
            cont = True
        return cont
        # preparation files

    def run_gpumd_task(self):
        try:
            self.dump_freq
        except AttributeError:
            # 捕获变量不存在时的错误
            self.make_gpumd_task_first = False
            self.make_gpumd_task()
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
        
    def post_gpumd_run(self):
        '''
        extract the information from gpumd run, this function is referenced by make_loop_train
        '''
        try:
            self.dump_freq
        except AttributeError:
            # 捕获变量不存在时的错误
            self.make_gpumd_task_first = False
            self.make_gpumd_task()
            self.make_gpumd_task_first = True
            dlog.info("remake the gpumd task")
        iter_dir = self.iter_dir
        threshold_low  = self.idata.get("uncertainty_threshold_low", 0.2)
        threshold_high = self.idata.get("uncertainty_threshold_high", 0.5)
        level = self.idata.get("uncertainty_level", 1)
        gpumd_dir = os.path.join(self.iter_dir, "01.gpumd")
        os.chdir(gpumd_dir)
        subprocess.run(["cat */dump.xyz > all.xyz"], shell = True, capture_output = True, text=True)
        subprocess.run(["cat */thermo.out > thermo.out"], shell = True, capture_output = True, text=True)
        dlog.info("-----start analysis the trajectory-----")
        atoms = read("all.xyz", index=":", format="extxyz")
        dlog.info(f"Totally {len(atoms)} structures.")
        model_devi:dict = self.idata["model_devi"][self.ii]
        max_candidate = model_devi.get("max_candidate")
        continue_from_old = self.idata.get("continue_from_old", False)
        if not max_candidate:
            max_candidate = self.idata.get("max_candidate")
        #检查效率，是重新复制粘贴快，不爆炸显存的话应该复制粘贴快吧
        if len(atoms) > max_candidate*2:
            # if os.path.isfile("random.json") and continue_from_old:
            #     with open("random.json", "r") as json_file:
            #         random_indices = json.load(json_file)
            # else:
            random_indices = random.sample(range(len(atoms)),max_candidate*2)
            random_indices.sort()
            # random_indices = [int(i) for i in random_indices]
                # with open("random.json", "w") as json_file:
                #     json.dump(random_indices,json_file)
        else:
            random_indices = range(len(atoms))
        # list=[]
        frame_index={
            "accurate":[],
            "candidate":[],
            "failed":[]
        }
        nep_dir = os.path.join(self.iter_dir, "00.nep")
        calculator_fs = glob(f"{nep_dir}/**/nep.txt")
        calculators = [NEP(calculator_fs[i]) for i in range(len(calculator_fs))]
        
        def get_force(atom:Atom,calculator):
            atom._calc=calculator
            return atom.get_forces()
        candidate_but_failed_index = []
        candidate_but_failed_atom_index = []
        # if os.path.isfile("frame_index.json") and continue_from_old:
        #     with open('frame_index.json', 'r', encoding='utf-8') as file:
        #         frame_index = json.load(file)
        # else:
        for ii in random_indices:
        # for ii in tqdm(random_indices):
            forceslist=[get_force(atoms[ii],calculator) for calculator in calculators]
            avg_force = np.average(forceslist,axis=0)
            deltaforcesquare = np.sqrt(np.mean(np.square([(forceslist[i] - avg_force) for i in range(len(forceslist))]).sum(axis=2),axis=0))
            # 先计算绝对值再mean
            absforce_avg = np.mean(np.sqrt(np.square(forceslist).sum(axis=2)),axis=0)
            relative_error = (deltaforcesquare/(absforce_avg+level))
            uncertainty = np.max(relative_error)
            corr_deltaforcesquare = uncertainty * (absforce_avg[np.argmax(relative_error)]+level)
            corr_absforce_avg = absforce_avg[np.argmax(relative_error)]

            if uncertainty < threshold_low:
                # counter["accurate"]+=1
                frame_index["accurate"].append([ii,uncertainty,corr_deltaforcesquare,corr_absforce_avg])
            elif uncertainty > threshold_high:
                # counter["failed"]+=1
                frame_index["failed"].append([ii,uncertainty,corr_deltaforcesquare,corr_absforce_avg])
            else:
                # counter["candidate"]+=1
                shortest_distance = get_shortest_distance(atoms[ii],candidate_but_failed_atom_index)
                if shortest_distance < 0.6 or absforce_avg.any() > 75:
                    frame_index["failed"].append([ii,uncertainty,corr_deltaforcesquare,corr_absforce_avg])
                    candidate_but_failed_index.append(ii)
                else:
                    frame_index["candidate"].append([ii,uncertainty,corr_deltaforcesquare,corr_absforce_avg])
        if candidate_but_failed_index:
            dlog.warning(f"candidate_but_failed_index:{candidate_but_failed_index}") #,candidate_but_failed_atom_index:{candidate_but_failed_atom_index[random_indices.index(candidate_but_failed_index[0])]}")
            atoms_candidate_but_failed = [atoms[i] for i in candidate_but_failed_index]
            write(f"candidate_but_failed.pdb",atoms_candidate_but_failed)
        original_failed_ratio = len(frame_index["failed"])/(len(random_indices))
        if len(frame_index["candidate"]) > max_candidate:
            selected_indices = random.sample(frame_index["candidate"], max_candidate)
            selected_indices.sort()
            # 将选中的帧保留在candidate中，其它的移到failed
            # selected_indices.sort()
            frame_index["candidate"] = selected_indices
            frame_index["failed"].extend([ii for ii in frame_index["candidate"] if ii not in selected_indices])


        assert(os.path.isabs(iter_dir))
        accurate_ratio = len(frame_index["accurate"])/(len(random_indices))
        dlog.info(f"accurate:{len(frame_index['accurate'])}, candidate:{len(frame_index['candidate'])}, failed:{len(frame_index['failed'])}, accurate_ratio:{accurate_ratio},original_failed:{original_failed_ratio}")
        
        label_dir = os.path.join(iter_dir, "02.label")
        os.makedirs(label_dir, exist_ok=True)
        # os.chdir(label_dir)

        if len(frame_index["candidate"]) == 0:
            # return True
            dlog.info("no candidate frames")
            raise ValueError("there are no candidate frames, maybe you can change the ensemble file")
        dlog.info("-----finish analysis the trajectory-----")


        # dlog.info(f"frame_index[\"candidate\"]]:{frame_index['candidate']}")
        candidate_info = frame_index["candidate"]
        # print(candidate)
        uncertainty = [iterm[1] for iterm in candidate_info]
        index = [iterm[0] for iterm in candidate_info]
        traj_selected = [atoms[i] for i in index]
        shortest_distance = [get_shortest_distance(atom) for atom in traj_selected]
        thermo = np.loadtxt("thermo.out")
        property_needed = thermo[:,[0,3]]
        property_needed = property_needed[index]
        self.frames_pertask = len(thermo)/self.model_devi_task_numbers
        remapped_index = [i%self.frames_pertask for i in index]
        property_needed = np.insert(property_needed, 0, remapped_index, axis=1)
        property_needed = np.insert(property_needed, property_needed.shape[1], shortest_distance, axis=1)
        property_needed = np.insert(property_needed, property_needed.shape[1], uncertainty, axis=1)
        sorted_indices = np.argsort(property_needed[:, 0])
        # 根据索引重新排序整个数组
        property_needed = property_needed[sorted_indices]
        with open("frame_index.json", "w") as json_file:
            json.dump(frame_index, json_file, indent=4)
        np.savetxt("candidate.txt",property_needed,fmt = "%12d %12.2f %12.2f %12.2f %12.2f", header = "%11s %-14s %-14s %-14s %-14s"%("index", "temperature", "pressure", "shortest_distance", "uncertainty"))
        os.remove("thermo.out")
        os.remove("all.xyz")

        atoms = [atoms[i] for i in index]
        write_extxyz(os.path.join(label_dir, "candidate.xyz"), atoms)
        del calculators
        del calculator_fs
        del atoms
        empty_cache()

    def make_label_task(self):
        label_engine = self.idata.get("label_engine","mattersim")
        if label_engine == "mattersim":
            self.make_mattersim_task()
        elif label_engine == "vasp":
            self.make_vasp_task()
        else:
            raise NotImplementedError(f"The label engine {label_engine} is not implemented")        

    def make_mattersim_task(self,pot_file:Optional[str]=None):
        """
        change the calculator to mattersim, and write the train.xyz
        """

        # 将工作目录设置为iter_dir下的02.label文件夹
        iter_dir = self.iter_dir
        work_dir = os.path.join(iter_dir, "02.label")
        train_ratio = self.idata.get("training_ratio", 0.8)
        os.chdir(work_dir)
        atoms = read("candidate.xyz", index=":", format="extxyz")
        self.pot_file = self.idata.get("pot_file")
        # 创建MatterSimCalculator对象，用于计算原子能量
        Nepactive.run_mattersim(pot_file=pot_file,train_ratio=train_ratio)

    @classmethod
    def run_mattersim(cls,pot_file:str,train_ratio:float=0.8):
        calculator = MatterSimCalculator(load_path=self.pot_file,device="cuda")
        traj = Trajectory('candidate.traj', mode='a')

        def change_calc(atom:Atom):
            atom._calc=calculator
            atom.get_potential_energy()
            traj.write(atom)
            return atom
        # 对每个原子调用change_calc函数，将计算结果写入Trajectory对象
        # atoms = [change_calc(atoms[i]) for i in tqdm(range(len(atoms)))]    
        atoms = [change_calc(atoms[i]) for i in range(len(atoms))]    
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
        write_extxyz("iter_train.xyz",train)
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
