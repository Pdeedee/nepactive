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

nvt_template = """
from ase.io import  read,write
from ase import Atoms,units
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
import numpy as np
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter
from nepactive.nphugo import NPHugo, MTTK
from ase.io import  read,write
from mattersim.forcefield import MatterSimCalculator
from ase.md.nvtberendsen import NVTBerendsen
calculator=MatterSimCalculator(device="cuda")
atoms = read("{structure}")
# 使用元素符号排序原子
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
opt = LBFGS(atoms)
opt.run(fmax=0.05,steps=40)
steps = 2000
write("opt.pdb",atoms)
temperature_K = 300
MaxwellBoltzmannDistribution(atoms,temperature_K=temperature_K)
traj = Trajectory('out.traj', 'w', atoms)
# pfactor= 100 #120
timestep = 0.2 * units.fs
dyn = NVTBerendsen(atoms, timestep=0.1*units.fs, temperature_K=300*units.kB, taut=0.5*1000*units.fs)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval=10)
dyn.attach(traj.write, interval=10)
dyn.run(steps)

"""

stable_nphugo_template = """
from ase.io import  read,write
from ase.md import MDLogger
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
import numpy as np
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter
from nepactive.nphugo import NPHugo, MTTK
from ase.io import  read,write
from mattersim.forcefield import MatterSimCalculator
calculator=MatterSimCalculator(device="cuda")
atoms = read("../structure/stable.pdb")
# 使用元素符号排序原子
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
ucf = UnitCellFilter(atoms,hydrostatic_strain=True)
opt = LBFGS(ucf)
opt.run(fmax=0.05,steps=40)
steps = {steps}
write("opt.pdb",atoms)
temperature_K = 300
MaxwellBoltzmannDistribution(atoms,temperature_K=temperature_K)
traj = Trajectory('out.traj', 'w', atoms)
# pfactor= 100 #120
timestep = 0.2 * units.fs
e0 = {e0}
p0 = 1*units.GPa
v0 = {v0}
pressure = {pressure} * units.GPa
dyn = NPHugo(atoms, e0 = e0, p0 = p0, v0=v0, p_stop=pressure, timestep=timestep, tchain=3, pchain=3)#, pfreq=0.001)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval=10)
dyn.attach(traj.write, interval=10)
dyn.run(steps)
"""

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
        dlog.info(f"StableRun: {self.idata}")

    def calculate_properties(self):
        dlog.info(f"start calculating properties")
        self.mass = self.atoms.get_masses().sum()
        self.volume = self.atoms.get_volume()
        self.energy = self.atoms.get_potential_energy()
        self.p0 = self.atoms.get_stress(voigt=False).trace()/3
        self.rho = self.mass/self.volume
        os.chdir(self.work_dir)
        os.makedirs("structure",exist_ok=True)
        os.chdir("structure")
        if not os.path.exists("task_finished"):
            # traj = Trajectory("out.traj", "w", self.atoms)
            # MaxwellBoltzmannDistribution(self.atoms,temp=300 * units.kB)
            # dyn = NVTBerendsen(self.atoms, timestep=0.2*units.fs, temperature_K=300*units.kB, taut=0.5*1000*units.fs)
            nvt_pyfile = nvt_template.format(structure=self.struc_file_name)
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
        self.atoms = read("out.traj", index = -1)
        self.energy = self.atoms.get_total_energy()
        self.p0 = self.atoms.get_stress(voigt=False).trace()/3



        # self.struc_num = struct_num

    def run(self):
        """
        根据self.atoms的原子种类生成稳定的产物初始结构，需要指定生成多少个
        """
        self.calculate_properties()
        dlog.info(f"start stable run")
        struct_dirs = []
        molecule_dicts = []
        for ii in range(self.struc_num):
            os.chdir(self.work_dir)
            os.makedirs(f"struc.{ii:03d}",exist_ok=True)
            struc_dir = os.path.abspath(f"struc.{ii:03d}")
            struct_dirs.append(struc_dir)
            os.chdir(struc_dir)
            
            molecule_dict = self.make_molecule_dict()
            os.makedirs("structure",exist_ok=True)
            os.chdir("structure")
            if not os.path.isfile("stable.pdb"):
                os.system("cp /home/liuzf/scripts/packmol/*pdb .")
                self.make_structure(molecule_dict,name="stable.pdb")
                molecule_dicts.append(molecule_dict)      
            os.chdir(struc_dir)      
            self.make_tasks()
        
        os.chdir(self.work_dir)
        self.run_tasks()

    def run_tasks(self):
        """
        运行任务
        """
        os.chdir(self.work_dir)
        task_dirs = glob("*/task.*")
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
        return molecules_dict

    def make_tasks(self):
        """ 
        根据压力生成任务
        """
        work_dir = os.getcwd()
        task_dirs = []
        steps = self.idata.get("steps",400000)
        for ii,_ in enumerate(self.pressure_list):
            os.chdir(work_dir)
            task_dir = os.path.join(work_dir,f"task.{ii:03d}")
            task_dirs.append(task_dir)
            os.makedirs(task_dir,exist_ok=True)
            os.chdir(task_dir)
            py_file = stable_nphugo_template.format(e0=self.energy, v0=self.volume, pressure=self.pressure_list[ii], steps = steps)
            with open("ensemble.py","w",encoding='utf-8') as f:
                f.write(py_file)
        
    def post_process(self):
        os.chdir(self.work_dir)
        task_dirs = glob("*/task.*")
        task_dirs = [os.path.abspath(task_dir) for task_dir in task_dirs]
        task_dirs.sort()
        thermos = []
        thermo_averages = []
        for task_dir in task_dirs:
            thermo = np.loadtxt(os.path.join(task_dir,"md.log"))
            thermo = thermo[int(0.2*len(thermo)):int(0.9*len(thermo)),:]
            thermos.append(thermo)
            thermo_average = np.average(thermo,axis=0)
            thermo_averages.append(thermo_average)
        np.savetxt("thermo.txt",np.array(thermo_averages))
