from ase import Atoms
from ase.io import read,write
from typing import List
import numpy as np
from mattersim.forcefield import MatterSimCalculator
from ase.optimize import LBFGS
from nepactive import dlog
from nepactive.template import model_devi_template
import subprocess
import os
from glob import glob
from sympy import N
import numbers

def select_by_force(traj:List[Atoms],threshold:float=50):
    """select the frames with max force larger than threshold"""
    forces = [traj[i].get_forces() for i in range(len(traj))]
    maxforces = [np.max(forces[i]) for i in range(len(forces))]
    index = [i for i in range(len(maxforces)) if maxforces[i] < threshold]
    traj = [traj[i] for i in index]
    return traj


def small_opt(file:str):
    atoms = read(file)
    calculator=MatterSimCalculator(device="cuda")
    atoms._calc = calculator
    opt = LBFGS(atoms)
    opt.run(fmax=0.05,steps=200)
    write(f"{file.split('.')[0]}_opt.pdb",atoms)

#shock velocity calculation

from sympy import symbols, Eq, solve
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams.update({'axes.linewidth': 2, 'axes.edgecolor': 'black'})
# 设置全局字体加粗
plt.rcParams.update({
    'font.weight': 'bold',  # 全局字体加粗
    'axes.labelweight': 'bold',  # 坐标轴标签加粗
    'axes.titleweight': 'bold',  # 标题加粗
    'axes.linewidth': 2,  # 设置坐标轴边框线宽
    'axes.edgecolor': 'black'  # 设置坐标轴边框颜色
})
from cycler import cycler
default_cycler = (cycler(color=['b','r', 'g', 'y']) +
                  cycler(linestyle=['-', '--', ':', '-.']))
plt.rc('axes', prop_cycle=default_cycler)
def find_tangent_slope(a, b, c):
    # 定义变量
    x0 = symbols('x0')

    # 抛物线方程 y = ax^2 + bx + c
    # 切线方程 y - (ax0^2 + bx0 + c) = (2ax0 + b)(x - x0)
    # 其中 x0 为切点的横坐标，(1, 0) 是切线上的点
    y_tangent = (2*a*x0 + b)*(1 - x0) + (a*x0**2 + b*x0 + c)
    # 给定条件：切线经过点 (1, 0)
    tangent_equation = Eq(y_tangent, 0)
    # 解方程得到切点 x0
    solutions = solve(tangent_equation, x0)
    if solutions:
        x0_value = solutions[0]  # 假设我们取第一个解
        # 计算该点处的切线斜率
        slope = 2*a*x0_value + b
    else:
        raise ValueError("没有满足条件的切线。")
    # slope, x0 = find_tangent_slope(a, b, c)

    return slope, x0_value

def fit_quadratic(x, y):
    """
    给定 x 和 y 的数据点，拟合二次方程 y = ax^2 + bx + c
    返回拟合的系数 (a, b, c) 和拟合后的 y 值。
    """
    # 使用 numpy.polyfit 拟合二次方程，degree=2 表示二次多项式
    coefficients = np.polyfit(x, y, 2)
    
    # 获取拟合后的系数 (a, b, c)
    a, b, c = coefficients
    
    # 使用 np.polyval 生成拟合曲线的 y 值
    y_fit = np.polyval(coefficients, x)
    
    # 打印拟合的二次方程
    print(f"拟合得到的二次方程: y = {a}x^2 + {b}x + {c}")
    
    return a, b, c

def shock_calculate(volume,pressure,v0,rho):
    """
    根据给定的体积和压力数据，计算冲击速度。
    """

    rel_volume = volume/v0   #1
    pressure = np.abs(pressure)
    a,b,c = fit_quadratic(rel_volume, pressure)
    slope, x0 = find_tangent_slope(a, b, c)
    y0 = np.abs(slope * (x0 - 1))
    x = np.arange(0,1.1,0.01)

    y = np.abs(a * x**2 + b * x + c)
    # print(f"x,y:{x,y}")
    plt.plot(x, y)
    try:
        slope = float(slope)
        y_tangent = np.abs(slope * (x-1))
        plt.plot(x, y_tangent, label=f"y = {slope:8.2f}(x - 1)")
    except TypeError:
        slope = None
        dlog.error(f"TypeError: {slope}, the fitting is failed")


    plt.scatter(rel_volume, pressure, color='red')
    plt.xlim(0.5, 1)
    plt.ylim(0,int(max(pressure)*1.2))
    plt.xlabel('Relative Volume')
    plt.ylabel('Pressure (GPa)')
    plt.legend()

    if slope:
        slope = np.float64(slope)
        shock_vel = np.sqrt(abs(slope)/rho)
        plt.scatter(x0, y0, color='red', label="(1, 0)",marker="x")
        plt.annotate(f'(x0, y0):({x0:.2f},{y0:.2f})\ndetonation_vel:{shock_vel:.03f}km/s\nrho:{rho}', xy=(x0, y0), xytext=(0.8, max(pressure)),
                    fontsize=12, color='red')
        dlog.info(f"slope:{slope:.2f}")
        dlog.info(f"shock_vel:{shock_vel:.3f}")
        dlog.info(f"x0_num:{float(x0)}")
        dlog.info(f"y0_num:{float(y0)}")
    else:
        shock_vel = 0
        x0 = 0
        y0 = 0
    
    plt.savefig("shock_vel.png",dpi=300)


    plt.close()
    return shock_vel,x0,y0

def run_gpumd_task(work_dir:str=None,gpu_available:List[int]=None,task_per_gpu:int=1):
    """
    搜索当前目录文件夹中的task文件，并将其分配到可用的GPU上运行gpumd
    """
    tasks = glob("**/task.*", recursive=True)
    if not work_dir:
        work_dir = os.getcwd()

    if not tasks:
        raise RuntimeError(f"No task files found in {work_dir}")
    
    task_per_job = task_per_gpu * len(gpu_available)
    jobs = []
    for job_id in range(task_per_job):
        jobs.append([tasks[i] for i in range(0, len(tasks)) if (i % task_per_job) == job_id])
    job_list = []
    for index,job in enumerate(jobs):
        # text = ""
        # for task in job:
        #     text += model_devi_template.format(work_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd",
        #                                       task_dir = task)
        text = "".join([model_devi_template.format(work_dir=work_dir, task_dir=task) for task in job])
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
        gpu_id = gpu_available[index%len(gpu_available)]
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
            raise RuntimeError(f"One or more processes failed. Check the log ({work_dir}/{log_file}) files for details.")
        else: 
            dlog.info(f"gpumd run successfully. Log saved at: {work_dir}/{log_file}")

def compute_volume_from_thermo(thermo:np.ndarray):
    num_columns = thermo.shape[1]
    if num_columns == 12:
        # 向量化计算
        volume = np.abs(np.prod(thermo[:, 9:12], axis=1))
        return np.column_stack((thermo[:, [0,1,2,3]], volume))
    elif num_columns == 18:
        # 向量化计算叉积和点积
        a = thermo[:, 9:12]
        b = thermo[:, 12:15]
        c = thermo[:, 15:18]
        # 计算 b × c
        cross = np.cross(b, c)
        # 计算 a · (b × c)
        volume = np.abs(np.sum(a * cross, axis=1))
        return np.column_stack((thermo[:, [0,1,2,3]], volume))
    else:
        raise ValueError("thermo file has wrong format")