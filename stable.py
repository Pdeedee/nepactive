from nepactive.nphugo import MTTK, NPHugo
from nepactive.random_stable import solve_molecular_distribution
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from ase import Atoms,units
import numpy as np
import os
from collections import Counter
from nepactive.packmol import make_structure
import numpy as np
from glob import glob
import subprocess
from nepactive import dlog
from ase.io import read,write
from nepactive.nphugo import MTTK
from ase.io.trajectory import Trajectory
from mattersim.forcefield import MatterSimCalculator
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
from ase.build import make_supercell
from nepactive.template import nvt_pytemplate,nphugo_pytemplate,nphugo_template,shock_test_template
from nepactive.plt import ase_plt,gpumdplt
from nepactive.tools import shock_calculate,run_gpumd_task,compute_volume_from_thermo
from ase.io.extxyz import write_extxyz



class StableRun:
    def __init__(self, idata:dict):
        self.idata = idata
        self.struc_file_name = os.path.abspath(self.idata.get("structure"))
        self.atoms = read(self.struc_file_name)
        calculator=MatterSimCalculator(device="cuda")
        self.atoms._calc = calculator
        self.struc_num = self.idata.get("struc_num")
        self.work_dir = os.getcwd()
        self.pressure_list = self.idata.get("pressures",[20,25,30,35,40,45])
        cell = self.atoms.get_cell()
        cell_complete = self.atoms.get_cell(complete=True)
        self.analyze_range = self.idata.get("analyze_range",[0.5,1])
        self.molecule_dicts = []
        self.r_rho = self.idata.get("rho",None)
        dlog.info(f"cell: {cell},cell_complete: {cell_complete}")
        dlog.info(f"StableRun: {self.idata}")

    def calculate_properties(self):
        dlog.info(f"start calculating properties")
        self.mass = self.atoms.get_masses().sum()
        self.volume = self.atoms.get_volume()
        self.energy = self.atoms.get_potential_energy()
        self.p0 = self.atoms.get_stress(voigt=False).trace()/3
        self.rho0 = self.mass/self.volume / units.kg * units.m**3/1000
        os.chdir(self.work_dir)
        if os.path.exists("properties.txt"):
            property = np.loadtxt("properties.txt",encoding="utf-8")
            self.rho0 = property[0]
            self.energy = property[1]
            self.p0 = property[2]
            self.volume = property[3]
            dlog.info(f"find properties.txt, rho: {self.rho0}, energy: {self.energy}, p0: {self.p0}, volume: {self.volume}")
        else:    
            os.makedirs("structure",exist_ok=True)
            os.chdir("structure")
            if not os.path.exists("task_finished"):
                nvt_pyfile = nvt_pytemplate.format(structure=self.struc_file_name,temperature = 300, steps = 2000)
                python_interpreter = self.idata.get("python_interpreter")
                with open("ensemble.py","w") as f:
                    f.write(nvt_pyfile)
                env = os.environ.copy()
                # env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                with open("log", 'w') as log:
                    process = subprocess.Popen(
                            [python_interpreter, "ensemble.py"], 
                            stdout=log, 
                            stderr=subprocess.STDOUT,
                            env=env
                        )
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("NVT ensemble run failed")
                os.system("touch task_finished")
                dlog.info(f"init run for energy task finished")
            self.atoms:Atoms = read("out.traj", index = -1)
            self.energy = self.atoms.get_total_energy()
            self.p0 = -self.atoms.get_stress(voigt=False).trace()/3/units.GPa
            property = [self.rho0,self.energy,self.p0,self.volume]
            os.chdir(self.work_dir)
            # dlog.info(f"properties: {property}")
            format = "%12.2f "*4
            # dlog.info(f"format: {format},array: {np.array(property)}")
            # print(f"properties: {property}")
            np.savetxt("properties.txt",np.array(property).reshape(1, -1),fmt=format,encoding="utf-8")
        if self.r_rho is None:
            self.r_rho = self.rho0
            # dlog.info(f"rho is None, set rho = {self.rho0}")
            self.r_v = self.volume
            dlog.info(f"rho is None, set rho = {self.r_rho}, volume is not changed")
        else:
            self.r_v = self.rho0 / self.r_rho * self.volume
            dlog.info(f"rho is set to {self.r_rho}, volume is changed from {self.volume} to {self.r_v}")

    def run(self):
        """
        根据self.atoms的原子种类生成稳定的产物初始结构，需要指定生成多少个
        """
        self.pot = self.idata.get("pot","mattersim")
        if self.pot == "nep":
            self.nep = self.idata.get("nep",None)
            assert self.nep is not None, "nep is None"

        dlog.info(f"StableRun: {self.idata}")
        self.calculate_properties()
        dlog.info(f"start stable run")

        self.make_tasks()
        os.chdir(self.work_dir)
        self.run_tasks()
        self.post_process()

    def make_tasks(self):
        """
        生成任务
        """
        os.chdir(self.work_dir)
        if self.pot == "mattersim":
            self.make_mattersim_tasks()
        elif self.pot == "nep":
            self.make_gpumd_tasks()
        else:
            raise ValueError("Unknown pot")

    def run_tasks(self):
        if self.pot == "mattersim":
            self.run_pytasks()
        elif self.pot == "nep":
            self.run_gpumd_tasks()
        else:
            raise ValueError("Unknown pot")

    def make_mattersim_tasks(self):
        struc_dirs = []
        for ii in range(self.struc_num):
            os.chdir(self.work_dir)
            os.makedirs(f"struc.{ii:03d}",exist_ok=True)
            struc_dir = os.path.abspath(f"struc.{ii:03d}")
            struc_dirs.append(struc_dir)
            os.chdir(struc_dir)
            self.make_preparations()

        original_make = self.idata.get("original_make",True)
        if original_make:
            os.chdir(self.work_dir)
            os.makedirs("original",exist_ok=True)
            struc_dir = os.path.abspath("original")
            struc_dirs.append(struc_dir)
            os.chdir(struc_dir)
            atoms = read(f"{self.work_dir}/POSCAR")
            os.makedirs("structure",exist_ok=True)
            write(f"structure/stable.pdb",atoms)
            self.make_preparations()

    def make_preparations(self):
        new = False
        max_try = 0

        struc_dir = os.getcwd()

        error = 1
        if not os.path.exists("structure/stable.pdb"):           
            while (not new) or (error != 0):
                molecule_dict,error = self.make_molecule_dict()
                if molecule_dict not in self.molecule_dicts:
                    self.molecule_dicts.append(molecule_dict)
                    new = True
                    dlog.info(f"molecule_dict {molecule_dict} is new")
                else:
                    max_try += 1
                    dlog.info(f"molecule_dict {molecule_dict} already exists, try again")
                    if max_try > 10:
                        raise ValueError("Too many tries, please check the molecule_dict")

            os.makedirs("structure",exist_ok=True)
            os.chdir("structure")
            
            if not os.path.isfile("stable.pdb"):
                os.system("cp /home/liuzf/scripts/packmol/*pdb .")
                self.make_structure(molecule_dict,name="stable.pdb")
                dlog.info(f"make structure for {molecule_dict}")

        os.chdir(struc_dir)
        if not os.path.exists("task.000"):
            self.make_pytasks()

    def run_pytasks(self):
        """
        运行任务
        """
        os.chdir(self.work_dir)
        task_dirs = glob("*/**/task.*",recursive=True)
        task_dirs = [os.path.abspath(task_dir) for task_dir in task_dirs]
        if not task_dirs:
            raise ValueError("No task dir found")
        task_dirs.sort()
        
        task_per_gpu = self.idata.get("task_per_gpu",4)
        gpu_available = self.idata.get("gpu_available",[0,1,2,3])
        tasknum_pertime = task_per_gpu * len(gpu_available)
        
        # 将任务分成多个批次，每个批次最多运行tasknum_pertime个任务
        batches = [task_dirs[i:i+tasknum_pertime] for i in range(0, len(task_dirs), tasknum_pertime)]
        dlog.info(f"Total {len(task_dirs)} tasks, {len(batches)} batches, {tasknum_pertime} tasks per batch")
        for batch in batches:
            processes = []
            original_dir = os.getcwd()
            
            # 启动当前批次的所有任务
            for index, task_dir in enumerate(batch):
                os.chdir(original_dir)  # 返回原始目录
                
                if os.path.exists(os.path.join(task_dir, "task_finished")):
                    continue
                os.chdir(task_dir)
                
                python_interpreter = self.idata.get("python_interpreter")
                gpu_id = gpu_available[index % len(gpu_available)]
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
                log_file = os.path.join(task_dir, 'log')
                with open(log_file, 'w') as log:
                    process = subprocess.Popen(
                        [python_interpreter, "ensemble.py"], 
                        stdout=log, 
                        stderr=subprocess.STDOUT,
                        env=env
                    )
                    processes.append((process, task_dir))
            
            if not processes:
                continue

            # 等待当前批次的所有进程完成
            os.chdir(original_dir)  # 返回原始目录
            for process, task_dir in processes:
                process.wait()
                if process.returncode != 0:
                    dlog.error(f"Process failed. Check the log at: {os.path.join(task_dir, 'log')}")
                    raise RuntimeError(f"Process failed. Check the log at: {os.path.join(task_dir, 'log')}")
                else:
                    os.system(f"touch {task_dir}/task_finished")
                    dlog.info(f"Process completed successfully. Log saved at: {os.path.join(task_dir, 'log')}")

    def make_structure(self,molecule_dict:dict,name):
        """
        根据molecule_dict生成结构
        """
        dlog.info(f"start make structure according to {molecule_dict}")
        lattice = np.power(self.volume, 1/3)
        cell = [lattice,lattice,lattice]

        make_structure(molecules=molecule_dict, cell=cell, name=name)

    def make_molecule_dict(self):
        """
        生成单个结构
        """
        symbols = self.atoms.get_chemical_symbols()
        element_counts = Counter(symbols)
        c =  element_counts['C']
        h =  element_counts['H']
        o =  element_counts['O']
        n =  element_counts['N']
        result = solve_molecular_distribution(c,h,o,n)
        max_try = 0
        while result["error"] != 0 and max_try < 10:
            result = solve_molecular_distribution(c,h,o,n)
            max_try += 1
        molecules_dict = result["solution"]
        dlog.info(f"molecule_dict: {molecules_dict},error: {result['error']}")
        return molecules_dict,result["error"]

    def make_pytasks(self):
        """ 
        根据压力生成任务
        """
        work_dir = os.getcwd()
        task_dirs = []
        steps = self.idata.get("steps",400000)
        self.total_time = steps * 0.2
        for ii,_ in enumerate(self.pressure_list):
            os.chdir(work_dir)
            task_dir = os.path.join(work_dir,f"task.{ii:03d}")
            task_dirs.append(task_dir)
            os.makedirs(task_dir,exist_ok=True)
            os.chdir(task_dir)
            dump_freq = self.idata.get("dump_freq",100)

            py_file = nphugo_pytemplate.format(structure = "../structure/stable.pdb",e0=self.energy,dump_freq = dump_freq,
                                                    v0=self.r_v, pressure=self.pressure_list[ii], steps = steps)
            with open("ensemble.py","w",encoding='utf-8') as f:
                f.write(py_file)

    def post_process(self):
        if self.pot == "mattersim":
            self.post_pyprocess()
        elif self.pot == "nep":
            self.post_nep_process()
        else:
            raise ValueError("Unknown pot")
    
    def post_nep_process(self):
        dlog.info("start post process")
        dlog.info(f"rho = {self.r_rho}")
        os.chdir(self.work_dir)
        task_dirs = glob("*/**/task.*",recursive=True)
        task_dirs = [os.path.abspath(task_dir) for task_dir in task_dirs]
        task_dirs.sort()
        thermos = []
        # thermo = None
        thermo_averages = []
        gpumd_steps = self.idata.get("gpumd_steps",400000)
        self.total_time = gpumd_steps * 0.2
        # self.r_v = self.rho0 / self.r_rho * self.volume
        for task_dir in task_dirs:
            os.chdir(task_dir)
            dlog.info(f"post process {task_dir}")
            thermo = np.loadtxt(os.path.join(task_dir,"thermo.out"),skiprows=1,encoding="utf-8")
            gpumdplt(total_time = self.total_time)
            # thermo = np.loadtxt("thermo.out")
            thermo_new = compute_volume_from_thermo(thermo)[:,[0,2,3,-1]]
            # frame_property = np.concatenate((property_list_np,thermo_new),axis=1)
            thermo = thermo_new[int(self.analyze_range[0]*len(thermo)):int(self.analyze_range[1]*len(thermo)),:]
            thermos.append(thermo)
            thermo_average = np.average(thermo,axis=0)
            thermo_averages.append(thermo_average)
        os.chdir(self.work_dir)
        format = "%12.2f"*4
        with open("thermo.txt",'w',encoding="utf-8") as file:
            np.savetxt(file,np.array(thermo_averages),encoding='utf-8',fmt=format,
                   header=f"Temperature[K]     Pot[eV]     Pressure[GPa]    V[Å^3]")
        dlog.info("saved thermo.txt")
        dlog.info("post process finished")

        thermos = np.loadtxt("thermo.txt",encoding="utf-8")
        shock_vels = []
        struc_dir = glob("struc.*")
        struc_dir = [os.path.abspath(struc) for struc in struc_dir]
        struc_dir.sort()
        for ii,struc in enumerate(struc_dir):
            os.chdir(struc)
            start_index = ii * len(self.gpumd_pressure_list)
            thermo_branch = thermos[start_index:start_index+len(self.gpumd_pressure_list),:]
            volume = thermo_branch[:,3]
            pressure = thermo_branch[:,2]
            dlog.info(f"volume: {volume}, pressure: {pressure}, v0: {self.r_v}, rho: {self.r_rho}, initial volume: {self.volume}")
            shock_vel = shock_calculate(volume=volume,pressure=pressure,v0=self.r_v, rho=self.r_rho)
            shock_vels.append(shock_vel)
        os.chdir(self.work_dir)
        with open("shock_vel.txt",'w') as file:
            np.savetxt(file,np.array(shock_vels),fmt="%12.2f")
        dlog.info("saved shock_vel.txt")
                
        
    def post_pyprocess(self):
        dlog.info("start post process")
        dlog.info(f"rho = {self.r_rho}")
        os.chdir(self.work_dir)
        task_dirs = glob("*/**/task.*",recursive=True)
        task_dirs = [os.path.abspath(task_dir) for task_dir in task_dirs]
        task_dirs.sort()
        thermos = []
        # thermo = None
        thermo_averages = []
        for task_dir in task_dirs:
            os.chdir(task_dir)
            dlog.info(f"post process {task_dir}")
            ase_plt()
            thermo = np.loadtxt(os.path.join(task_dir,"md.log"),skiprows=1,encoding="utf-8")
            thermo = thermo[int(self.analyze_range[0]*len(thermo)):int(self.analyze_range[1]*len(thermo)),:]
            thermos.append(thermo)
            thermo_average = np.average(thermo,axis=0)
            thermo_averages.append(thermo_average)
        os.chdir(self.work_dir)
        format = "%12.2f"*12
        with open("thermo.txt",'w',encoding="utf-8") as file:
            np.savetxt(file,np.array(thermo_averages),encoding='utf-8',fmt=format,
                   header=f"Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]        V[Å^3]      ---------------------- stress [GPa] -----------------------")
        dlog.info("saved thermo.txt")
        dlog.info("post process finished")

        thermos = np.loadtxt("thermo.txt",encoding="utf-8")
        shock_vels = []
        struc_dir = glob("struc.*")
        struc_dir = [os.path.abspath(struc) for struc in struc_dir]
        struc_dir.sort()
        for ii,struc in enumerate(struc_dir):
            os.chdir(struc)
            start_index = ii * len(self.pressure_list)
            thermo_branch = thermos[start_index:start_index+len(self.pressure_list),:]
            volume = thermo_branch[:,5]
            pressure = np.mean(thermo_branch[:,6:9],axis=1)
            shock_vel = shock_calculate(volume=volume,pressure=pressure,v0=self.r_v, rho=self.r_rho)
            shock_vels.append(shock_vel)
        os.chdir(self.work_dir)
        with open("shock_vel.txt",'w') as file:
            np.savetxt(file,np.array(shock_vels),fmt="%12.2f"*3)
        dlog.info("saved shock_vel.txt")
            
    def make_gpumd_tasks(self):
        os.chdir(self.work_dir)
        task_dirs = []
        struc_prefix = self.idata.get("struc_prefix",os.getcwd())
        strucs = self.idata.get("structure_files",[])
        # strucs = self.idata.get("structure_files",[])
        assert strucs != [], "structure_files is empty"
        if not os.path.isabs(strucs[0]):
            strucs = [os.path.join(struc_prefix,struc) for struc in strucs]
        self.task_dirs = []
        for ii,struc in enumerate(strucs):
            dlog.info(f"make gpumd tasks for {struc}")
            os.chdir(self.work_dir)
            os.makedirs(f"struc.{ii:03d}",exist_ok=True)
            os.chdir(f"struc.{ii:03d}")
            self.make_gpumd_tasks_for_struc(struc,ii)

    def make_gpumd_tasks_for_struc(self,struc,ii):
        work_dir = os.path.abspath(os.getcwd())
        self.gpumd_pressure_list = self.idata.get("gpumd_pressure_list",[20,25,30,35,40,45,50,60])
        gpumd_steps = self.idata.get("gpumd_steps",400000)
        dump_freq = gpumd_steps//2500
        dlog.info(f"r_v: {self.r_v}, r_rho: {self.r_rho}, rho0: {self.rho0}, v0: {self.volume}")
        # os.system("rm -rf task.*")
        for ii,_ in enumerate(self.gpumd_pressure_list):
            os.chdir(work_dir)
            task_dir = os.path.join(work_dir,f"task.{ii:03d}")
            os.makedirs(task_dir,exist_ok=True)
            os.chdir(task_dir)
            self.task_dirs.append(task_dir)

            atoms = read(struc)
            write_extxyz("model.xyz",atoms)
            os.system(f"ln -snf {self.nep} .")
            dlog.info(f"model.xyz and {self.nep} are linked")
            # dump_freq = self.idata.get("dump_freq",100)

            text = shock_test_template.format(
                time_step = 0.2,
                run_steps = gpumd_steps,
                dump_freq = dump_freq,
                replicate_cell = "1 1 1",
                pressure = self.gpumd_pressure_list[ii],
                e0 = self.energy,
                v0 = self.r_v,
                p0 = self.p0
            )
            with open("run.in","w",encoding='utf-8') as f:
                f.write(text)

    def run_gpumd_tasks(self):
        gpu_available = self.idata.get("gpu_available",[0,1,2,3])
        self.task_per_gpu = self.idata.get("task_per_gpu",1)
        run_gpumd_task(gpu_available=gpu_available, task_per_gpu=self.task_per_gpu)
