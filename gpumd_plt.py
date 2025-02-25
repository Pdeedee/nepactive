#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
def gpumdplt(dump_interval,time_step):
    # 检查命令行参数，获取初始体积
    #if len(sys.argv) < 2:
    #    raise ValueError("需要提供初始体积作为命令行参数！")
    # initial_volume = float(sys.argv[1])

    # 读取数据
    data = np.loadtxt('thermo.out')

    # dump_interval = 10  
    time = np.arange(0, len(data) * dump_interval * time_step / 1000, dump_interval * time_step/ 1000)

    # 读取数据列
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
        
        # 计算相对体积
        # relative_volume = volume / initial_volume
    else:
        raise ValueError("不支持的 thermo.out 文件列数。期望 12 或 18 列。")

    # 子图
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), dpi=100)

    # 温度
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
#else:
#    plt.show()

