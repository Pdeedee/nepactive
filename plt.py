#!/usr/bin/env python3
import sys
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

def gpumdplt(total_time,time_step=0.2):

    data = np.loadtxt('thermo.out')

    num_points = len(data)  # 数据点数
    total_time = total_time  # 总时间
    time = np.linspace(0, total_time, num_points, endpoint=False)/1000

    temperature = data[:, 0]
    kinetic_energy = data[:, 1]
    potential_energy = data[:, 2]
    pressure_x = data[:, 3]
    pressure_y = data[:, 4]
    pressure_z = data[:, 5]

    num_columns = data.shape[1]

    if num_columns == 12:
        box_length_x = data[:, 9]
        box_length_y = data[:, 10]
        box_length_z = data[:, 11]
        volume = np.abs(box_length_x * box_length_y * box_length_z)
        
    elif num_columns == 18:
        ax, ay, az = data[:, 9], data[:, 10], data[:, 11]
        bx, by, bz = data[:, 12], data[:, 13], data[:, 14]
        cx, cy, cz = data[:, 15], data[:, 16], data[:, 17]
        
        # 计算晶胞的体积（使用行列式公式）
        # 叉积 (b x c)
        bx_cy_bz = by * cz - bz * cy
        bx_cz_by = bz * cx - bx * cz
        bx_cx_by = bx * cy - by * cx
        
        # 点积 a · (b x c)
        volume = ax * bx_cy_bz + ay * bx_cz_by + az * bx_cx_by
        volume = np.abs(volume)  # 体积取绝对值
        
    else:
        raise ValueError("不支持的 thermo.out 文件列数。期望 12 或 18 列。")

    # 子图
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), dpi=100)

    # 温度
    # print(f"time = {time.shape}, temperature = {temperature.shape}")
    axs[0, 0].plot(time, temperature)
    axs[0, 0].set_title('Temperature')
    axs[0, 0].set_xlabel('Time (ps)')
    axs[0, 0].set_ylabel('Temperature (K)')

    # 势能与动能
    color_potential = 'tab:orange'
    color_kinetic = 'tab:green'
    axs[0, 1].set_title(r'$P_E$ vs $K_E$')
    axs[0, 1].set_xlabel('Time (ps)')
    axs[0, 1].set_ylabel('Potential Energy (eV)', color=color_potential)
    axs[0, 1].plot(time, potential_energy, color=color_potential)
    axs[0, 1].tick_params(axis='y', labelcolor=color_potential)

    axs_kinetic = axs[0, 1].twinx()
    axs_kinetic.set_ylabel('Kinetic Energy (eV)', color=color_kinetic)
    axs_kinetic.plot(time, kinetic_energy, color=color_kinetic)
    axs_kinetic.tick_params(axis='y', labelcolor=color_kinetic)

    # 压力
    axs[1, 0].plot(time, pressure_x, label='Px')
    axs[1, 0].plot(time, pressure_y, label='Py')
    axs[1, 0].plot(time, pressure_z, label='Pz')
    axs[1, 0].set_title('Pressure')
    axs[1, 0].set_xlabel('Time (ps)')
    axs[1, 0].set_ylabel('Pressure (GPa)')
    axs[1, 0].legend()

    # 相对体积
    axs[1, 1].plot(time, volume, label='Volume')
    axs[1, 1].set_title('Volume')
    axs[1, 1].set_xlabel('Time (ps)')
    axs[1, 1].set_ylabel('Volume')
    axs[1, 1].legend()

    plt.tight_layout()

    # 保存或显示图像
    #if len(sys.argv) > 2 and sys.argv[2] == 'save':
    plt.savefig('thermo.png')
    plt.close()


from pylab import *

def nep_plt(testplt=True):
    if testplt:
        prefix = 'test_'
        energy_file = 'energy_test.out'
        force_file = 'force_test.out'
        virial_file = 'virial_test.out'
    else:
        prefix = 'train_'
        energy_file = 'energy_train.out'
        force_file = 'force_train.out'
        virial_file = 'virial_train.out'
    loss = loadtxt('loss.out')
    loglog(loss[:, 1:6])
    loglog(loss[:, 7:9])
    xlabel('Generation/100')
    ylabel('Loss')
    legend(['Total', 'L1-regularization', 'L2-regularization', 'Energy-train', 'Force-train', 'Energy-test', 'Force-test'])
    tight_layout()
    plt.savefig('loss.png', dpi=300)  # 你可以修改文件名和文件格式
    plt.close()

    energy = loadtxt(energy_file)
    x_min = np.min(energy[:, :])
    x_max = np.max(energy[:, :])
    plot(energy[:, 1], energy[:, 0], '.')
    plot(linspace(x_min,x_max), linspace(x_min,x_max), '-')
    xlabel('DFT energy (eV/atom)')
    ylabel('NEP energy (eV/atom)')
    tight_layout()
    plt.savefig(f'{prefix}energy.png', dpi=300)  # 你可以修改文件名和文件格式
    plt.close()

    force = loadtxt(force_file)
    x_min = np.min(force[:, :])
    x_max = np.max(force[:, :])
    plot(force[:, 3:6], force[:, 0:3], '.')
    plot(linspace(x_min,x_max), linspace(x_min,x_max), '-',color='r')
    xlabel('DFT force (eV/A)')
    ylabel('NEP force (eV/A)')
    legend(['x direction', 'y direction', 'z direction'])
    tight_layout()
    plt.savefig(f'{prefix}force.png', dpi=300)  # 你可以修改文件名和文件格式
    plt.close()

    virial = loadtxt(virial_file)
    x_min = np.min(virial[:, :])
    x_max = np.max(virial[:, :])
    plot(virial[:, 6:11], virial[:, 0:5], '.')
    plot(linspace(x_min,x_max), linspace(x_min,x_max), '-',color='r')
    xlabel('DFT Virial (eV/A)')
    ylabel('NEP Virial (eV/A)')
    legend(['xx', 'yy', 'zz', 'xy', 'yz', 'zx'])
    tight_layout()
    plt.savefig(f'{prefix}virial.png', dpi=300)  # 你可以修改文件名和文件格式
    plt.close()

def ase_plt():
    data = np.loadtxt('md.log',skiprows=1,encoding='utf-8')
    num_points = len(data)  # 数据点数
    time = data[:, 0]

    temperature = data[:, 4]
    kinetic_energy = data[:, 3]
    potential_energy = data[:, 2]
    pressure_x = data[:, 6]
    pressure_y = data[:, 7]
    pressure_z = data[:, 8]
    volume = data[:, 5]

    # 子图
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), dpi=100)

    # 温度
    # print(f"time = {time.shape}, temperature = {temperature.shape}")
    axs[0, 0].plot(time, temperature)
    axs[0, 0].set_title('Temperature')
    axs[0, 0].set_xlabel('Time (ps)')
    axs[0, 0].set_ylabel('Temperature (K)')

    # 势能与动能
    color_potential = 'tab:orange'
    color_kinetic = 'tab:green'
    axs[0, 1].set_title(r'$P_E$ vs $K_E$')
    axs[0, 1].set_xlabel('Time (ps)')
    axs[0, 1].set_ylabel('Potential Energy (eV)', color=color_potential)
    axs[0, 1].plot(time, potential_energy, color=color_potential)
    axs[0, 1].tick_params(axis='y', labelcolor=color_potential)

    axs_kinetic = axs[0, 1].twinx()
    axs_kinetic.set_ylabel('Kinetic Energy (eV)', color=color_kinetic)
    axs_kinetic.plot(time, kinetic_energy, color=color_kinetic)
    axs_kinetic.tick_params(axis='y', labelcolor=color_kinetic)

    # 压力
    axs[1, 0].plot(time, pressure_x, label='Px')
    axs[1, 0].plot(time, pressure_y, label='Py')
    axs[1, 0].plot(time, pressure_z, label='Pz')
    axs[1, 0].set_title('Pressure')
    axs[1, 0].set_xlabel('Time (ps)')
    axs[1, 0].set_ylabel('Pressure (GPa)')
    axs[1, 0].legend()

    # 相对体积
    axs[1, 1].plot(time, volume, label='Volume')
    axs[1, 1].set_title('Volume')
    axs[1, 1].set_xlabel('Time (ps)')
    axs[1, 1].set_ylabel('Volume')
    axs[1, 1].legend()

    plt.tight_layout()

    # 保存或显示图像
    #if len(sys.argv) > 2 and sys.argv[2] == 'save':
    plt.savefig('thermo.png')
    plt.close()