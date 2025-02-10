import os
from nepactive import dlog
import shutil
import subprocess
from ase.io import read, write, Trajectory
from ase.io.extxyz import write_extxyz
from glob import glob
from ase import Atoms
import random
from typing import List
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

def parse_yaml(file):
    with open('in.yaml', 'r') as file:
        data = yaml.safe_load(file)
        # data = yaml.safe_load("in.yaml")
    return data

def record_iter(record, ii, jj):
    with open(record, "a") as frec:
        frec.write("%d %d\n" % (ii, jj))

def run():
    '''
    using different engines to make initial training data
    in.yaml need:
    init_engine: the engine to use for initial training data
    init_template_files: the template files to use for initial training data
    python_interpreter: the python interpreter to use for initial training data
    training_ratio: the ratio of training data to total data
    '''
    #change working directory
    idata:dict = parse_yaml("in.yaml")
    engine = idata.get("ini_engine","ase")
    python_interpreter=idata.get("python_interpreter")
    # work_dir = "."
    ini_frames = idata.get("ini_frames",10)
    template_files = idata.get("ini_template_files")
    training_ratio = idata.get("training_ratio")
    work_dir = os.getcwd()
    os.chdir(work_dir)
    os.makedirs("init",exist_ok=True)
    record = "record.nep"
    #init engine choice
    dlog.info(f"Initializing engine and make initial training data using {engine}")
    if not os.path.isfile(record):
        make_init_ase_run(work_dir = work_dir, idata = idata) 
        dlog.info("Extracting data from initial runs")
        make_data_extraction(work_dir = work_dir, idata = idata)
    make_loop_train(work_dir=work_dir, idata=idata)

def make_init_ase_run(work_dir: str, idata:dict):
    '''
    For the template file, the file name must be fixed form.
    Assumed that the working directory is already the correct directory.
    '''
    # Change the working directory to the init directory
    work_dir = f"{work_dir}/init"
    ase_ensemble_files:list[str] = idata.get("ini_ase_ensemble_files")
    python_interpreter:str = idata.get("python_interpreter")
    processes = []
    pot_file:str = idata.get("pot_file")
    # Make initial training task directories
    for index, file in enumerate(ase_ensemble_files):
        os.chdir(work_dir)
        task_name = f"task.{index:06d}"
        
        # Ensure the task directory is created
        task_dir = os.path.join(work_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)  # Create the task directory, if it doesn't exist
        
        # shutil.copy(file, task_dir)
        # os.symlink(pot_file, task_dir)
        os.system(f"ln -snf {pot_file} {task_dir}")
        os.system(f"ln -snf {file} {task_dir}")
        os.chdir(task_dir)
        
        # Get the basename of the file (e.g., "npt.py")
        basename = os.path.basename(file)

        # Open subprocess and redirect stdout and stderr to a log file
        log_file = os.path.join(task_dir, 'log')  # Log file path
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                [python_interpreter, basename], 
                stdout=log, 
                stderr=subprocess.STDOUT  # Combine stderr with stdout
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

def make_data_extraction(work_dir, idata:dict):
    '''
    extract data from initial runs, turn it into the gpumd format
    '''
    train:List[Atoms]=[]
    test:List[Atoms]=[]
    training_ratio:float = idata.get("training_ratio", 0.8)
    init_frames:int = idata.get("ini_frames", 100)
    # work_dir = f"{work_dir}/init"
    os.chdir(work_dir)
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
        dlog.warning(f"Too many frames to extract {init_frames} frames.")
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

# constants define
MaxLength = 70

def sepline(ch="-", sp="-"):
    r"""Seperate the output by '-'."""
    # if screen:
    #     print(ch.center(MaxLength, sp))
    # else:
    dlog.info(ch.center(MaxLength, sp))

def make_loop_train(work_dir, idata:dict):
    '''
    make loop training task
    '''
    
    os.chdir(work_dir)
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
    ii = -1
    numb_task = 4
    max_tasks = 10000
    while cont:
        ii += 1
        if ii < iter_rec[0]:
            continue
        iter_dir = os.path.abspath(os.path.join(work_dir,f"iter.{ii:06d}")) #the work path has been changed
        os.makedirs(iter_dir, exist_ok=True)
        iter_name = f"iter.{ii:06d}"
        # iter_name = make_iter_name(ii)
        # sepline(iter_name, "=")
        for jj in range(numb_task):
            if ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1]:
                continue
            task_name = "task %02d" % jj
            sepline(f"{iter_name} {task_name}", "-")
            if jj == 0:
                # log_iter("make_train", ii, jj)
                make_nep_train(work_dir = work_dir, ii=ii, iter_dir=iter_dir, idata = idata)
            elif jj == 1:
                # log_iter("run_train", ii, jj)
                cont = make_gpumd_run(iter_dir = iter_dir, idata = idata, ii = ii)
                # cont = make_model_devi(ii, jdata, mdata)
                if not cont:
                    break
            elif jj == 2:
                # log_iter("post_train", ii, jj)
                skip = post_gpumd_run(iter_dir=iter_dir, idata = idata)
            elif jj == 3:
                maker_label_run(skip = skip, iter_dir=iter_dir, idata = idata)
            else:
                raise RuntimeError("unknown task %d, something wrong" % jj)
            
            os.chdir(work_dir)
            record_iter(record, ii, jj)

def record_iter(record, ii, jj):
    with open(record, "a") as frec:
        frec.write("%d %d\n" % (ii, jj))

def make_nep_train(work_dir, ii, iter_dir, idata:dict):
    '''
    Train nep. 
    '''
    #ensure the work_dir is the absolute pathk
    global_work_dir = os.path.abspath(work_dir)
    work_dir = os.path.abspath(os.path.join(iter_dir, "00.nep"))
    pot_num = idata.get("pot_num", 4)
    init:bool = idata.get("init", True)
    #注意template的格式
    nep_template = idata.get("nep_template")
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
    # if init == True:
    #     # print(f"{os.path.isfile(os.path.join(global_work_dir, 'init/train.xyz'))}")
    #     files = files.extend(glob(os.path.join(global_work_dir, "init/train.xyz")))
    #     testfiles = testfiles.extend(glob(os.path.join(global_work_dir, "init/test.xyz")))
    #     if not files:
    #         raise ValueError("No files found to merge.")
    if ii > 0:
        for iii in range(ii):
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
        # with open("nep.in",'w',encoding='utf-8') as f:
        #     f.write(nep_template)
        #run nep
        log_file = os.path.join(task_dir, 'subprocess_log.txt')  # Log file path
        # with open(log_file, 'w') as log:
        #     process = subprocess.Popen(
        #         [f"CUDA_VISIBLE_DEVICES={jj}", "nep"], 
        #         stdout=log, 
        #         stderr=subprocess.STDOUT  # Combine stderr with stdout
        #     )
        #     processes.append((process, log_file))  # Store the process and log file
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(jj)
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                ["nep"],  # 程序名
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env  # 使用修改后的环境
            )
            processes.append((process, log_file))


    for process, log_file in processes:
        process.wait()  # Wait for all processes to complete
        # Check for errors using the return code
        if process.returncode != 0:
            dlog.error(f"Process failed. Check the log at: {log_file}")
            raise RuntimeError(f"One or more processes failed. Check the log file:({log_file}) for details.")
        else:
            dlog.info(f"Process completed successfully. Log saved at: {log_file}")

    # work_dir = os.path.abspath(f"{work_dir}/task.{ii:06d}")
    
def make_gpumd_run(iter_dir, idata:dict, ii):
    '''
    run gpumd, this function is referenced by make_loop_train
    '''
    # model_devi_jobs = idata["model_devi"]
    iter_index = ii

    # print(f"gpumd_templates:{gpumd_templates}")
    model_devi_jobs = idata["model_devi"]
    gpumd_dir = os.path.join(iter_dir, "01.gpumd")
    if iter_index > len(model_devi_jobs):
        cont = False
        return cont
    else:
        cont = True
    gpumd_templates:list[str] = idata["model_devi"][ii].get("gpumd_template")
    xyz_files = idata["model_devi"][ii].get("xyz_file")
    # ensure the work_dir is the right
    if not os.path.isabs(iter_dir):
        iter_dir = os.path.abspath(iter_dir)
        dlog.warning(f"iter_dir is not an absolute path, it is now {iter_dir}")
    os.makedirs(gpumd_dir, exist_ok=True)
    os.chdir(gpumd_dir)
    # preparation files
    processes = []
    for jj in range(len(gpumd_templates)):
        task_path = os.path.join(gpumd_dir, f"task.{jj:06d}")
        os.makedirs(task_path, exist_ok=True)
        os.chdir(task_path)
        if not os.path.isfile(f"{os.path.join(task_path, 'run.in')}"):
            shutil.copy(gpumd_templates[jj], task_path)
        if not os.path.isfile("model.xyz"):
            os.symlink(xyz_files, "model.xyz")
        if not os.path.isfile("nep.txt"):
            os.symlink(f"../../00.nep/task.000000/nep.txt", "nep.txt")
        log_file = os.path.join(task_path, 'log')  # Log file path
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(jj)
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                ["gpumd"],  # 程序名
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env  # 使用修改后的环境
            )
            processes.append((process, log_file))
        # with open(log_file, 'w') as log:
        #     process = subprocess.Popen(
        #         [f"CUDA_VISIBLE_DEVICES={jj}", "gpumd"], 
        #         stdout=log, 
        #         stderr=subprocess.STDOUT  # Combine stderr with stdout
        #     )
        # processes.append((process, log_file))  # Store the process and log file

    for process, log_file in processes:
        process.wait()  # Wait for all processes to complete
        # Check for errors using the return code
        if process.returncode != 0:
            dlog.error(f"Process failed. Check the log at: {log_file}")
            raise RuntimeError(f"One or more processes failed. Check the log ({log_file}) files for details.")
        else: 
            dlog.info(f"gpumd run successfully. Log saved at: {log_file}")
    assert(os.path.isabs(gpumd_dir))
    return cont

    
def post_gpumd_run(iter_dir, idata:dict):
    '''
    extract the information from gpumd run, this function is referenced by make_loop_train
    '''
    threshold_low = idata.get("uncertainty_threshold_low", 0.2)
    threshold_high = idata.get("uncertainty_threshold_high", 0.5)
    level = idata.get("uncertainty_level", 1)
    gpumd_dir = os.path.join(iter_dir, "01.gpumd")
    os.chdir(gpumd_dir)
    subprocess.run(["cat */dump.xyz > all.xyz"], shell = True, capture_output = True, text=True)
    dlog.info("-----start analysis the trajectory-----")
    atoms = read("all.xyz", index=":", format="extxyz")
    dlog.info("-----finish analysis the trajectory-----")
    #检查效率，是重新复制粘贴快，不爆炸显存的话应该复制粘贴快吧
    # list=[]
    
    counter = Counter()
    counter["accurate"]=0
    counter["candidate"]=0
    counter["failed"]=0
    frame_index={
        "accurate":[],
        "candidate":[],
        "failed":[]
    }
    nep_dir = os.path.join(iter_dir, "00.nep")
    calculator_fs = glob(f"{nep_dir}/**/nep.txt")
    calculators = [NEP(calculator_fs[i]) for i in range(len(calculator_fs))]

    def get_force(atom:Atom,calculator):
        atom._calc=calculator
        return atom.get_forces()

    for ii in tqdm(range(len(atoms))):
        forceslist=[get_force(atoms[ii],calculator) for calculator in calculators]
        avg_force = np.average(forceslist,axis=0)
        deltaforce = [(forceslist[i] - avg_force) for i in range(len(forceslist))]
        deltaforcesquare = np.sqrt(np.mean(np.square([(forceslist[i] - avg_force) for i in range(len(forceslist))]).sum(axis=2),axis=0))
        absforce_avg = np.sqrt(np.square(np.mean(forceslist,axis=0)).sum(axis=1))
        relative_error = (deltaforcesquare/(absforce_avg+level))
        uncertainty = np.max(relative_error)
        
        if uncertainty < threshold_low:
            counter["accurate"]+=1
            frame_index["accurate"].append(ii)
        elif uncertainty > threshold_high:
            counter["failed"]+=1
            frame_index["failed"].append(ii)
        else:
            counter["candidate"]+=1
            frame_index["candidate"].append(ii)
    with open("frame_index.json", "w") as json_file:
        json.dump(frame_index, json_file, indent=4)
    assert(os.path.isabs(iter_dir))
    dlog.info(f"accurate:{counter['accurate']}, candidate:{counter['candidate']}, failed:{counter['failed']}")
    
    label_dir = os.path.join(iter_dir, "02.label")
    os.makedirs(label_dir, exist_ok=True)
    os.chdir(label_dir)
    atoms = [atoms[i] for i in frame_index["candidate"]]
    write_extxyz(os.path.join(label_dir, "candidate.xyz"), atoms)
    if counter["accurate"] == 0:
        return True
    else:
        return False

def maker_label_run(skip, iter_dir, idata:dict):
    """
    change the calculator to mattersim, and write the train.xyz
    """
    if skip:
        raise KeyError("The candidate structure is zero")
    # 将工作目录设置为iter_dir下的02.label文件夹
    work_dir = os.path.join(iter_dir, "02.label")
    train_ratio = idata.get("training_ratio", 0.8)
    os.chdir(work_dir)
    # 读取candidate.xyz文件中的原子信息
    atoms = read("candidate.xyz", index=":", format="extxyz")
    # 创建MatterSimCalculator对象，用于计算原子能量
    calculator = MatterSimCalculator(load_path="mattersim-v1.0.0-1m",device="cuda")
    # path = "/workplace/liuzf/duanjw/ase/msst/n5/mattersim/1.7/vs102/msst.traj"
    # 创建Trajectory对象，用于存储计算结果
    traj = Trajectory('temporary.traj', mode='a')
    # atoms=read("msst.traj",index=-1)
    # atoms = read(path,index="1:-1:10000")
    # 定义一个函数，用于将原子能量计算结果写入Trajectory对象
    
    def change_calc(atom:Atom):
        atom._calc=calculator
        atom.get_potential_energy()
        traj.write(atom)
        return atom
    # 对每个原子调用change_calc函数，将计算结果写入Trajectory对象
    atoms = [change_calc(atoms[i]) for i in tqdm(range(len(atoms)))]    
    # 读取Trajectory对象中的原子信息
    atoms = read("temporary.traj",index=":")
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
    # 将原子信息写入train_iter.xyz文件
    write_extxyz("iter_train.xyz",train)
    write_extxyz("iter_test.xyz",test)
    os.remove("temporary.traj")

