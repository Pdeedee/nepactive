
msst_template = """
replicate       {replicate_cell}
potential		nep.txt

minimize sd 1.0e-6 1000

time_step	    {time_step}
velocity		300

ensemble        nvt_ber 300 300 200 

dump_thermo		{dump_freq}
dump_exyz       {dump_freq} 
run			    20000

dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
ensemble msst {shock_direction} {v_shock} qmass {qmass} mu {viscosity} tscale 0.01
run                         {run_steps}
"""

nvt_template = """
replicate       {replicate_cell}
potential		nep.txt
minimize sd 1.0e-6 1000

time_step	    {time_step}
velocity		{temperature}

ensemble        nvt_ber {temperature} {temperature} 200 
dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
run			    {run_steps}
"""

npt_template = """
potential		nep.txt
minimize sd 1.0e-6 10000

time_step	    {time_step}
velocity		{temperature}

ensemble        npt_mttk {temperature} {temperature} iso {pressure} {pressure} tperiod 200 pperiod 5000

dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
run			    {run_steps}
"""

nphugo_template = """
replicate       {replicate_cell}
potential		nep.txt
minimize sd 1.0e-6 1000

time_step		{time_step}
velocity		300
ensemble        nvt_ber 300 300 200 
dump_thermo		{dump_freq}
dump_exyz       {dump_freq}

run			  20000

dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
ensemble nphug iso {pressure} {pressure} e0 {e0} p0 {p0} v0 {v0} pperiod 1000

run                       {run_steps}
"""

shock_test_template = """
replicate       {replicate_cell}
potential		nep.txt

time_step		{time_step}
velocity		1000

dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
ensemble nphug iso {pressure} {pressure} e0 {e0} p0 {p0} v0 {v0} pperiod 4000

run                       {run_steps}
"""

model_devi_template = """
cd {work_dir}
cd {task_dir}
if [ ! -f task_finished ] ;then 
rm -f dump.xyz thermo.out log
gpumd > log 2>&1
if test $? -eq 0; then touch task_finished;fi
fi

"""

nep_in_template ="""
{nep_in_header}
lambda_1      0
zbl           2
version       4       # default
cutoff        5 4     # default
n_max         4 4     # default
basis_size    8 8     # default
l_max         4 2 0   # default
neuron        30      # default
lambda_e      1.0     # default
lambda_f      1.0     # default
lambda_v      0.1     # default
batch         1000     # default
population    50      # default
generation    {train_steps}  # default
 
"""

nvt_pytemplate = """
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
from ase.md.nvtberendsen import NVTBerendsen
from mattersim.forcefield import MatterSimCalculator
calculator=MatterSimCalculator(device="cuda")
atoms = read("{structure}")
# 使用元素符号排序原子
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
opt = LBFGS(atoms)
opt.run(fmax=0.05,steps=40)
steps = {steps}
write("opt.pdb",atoms)
temperature_K = {temperature}
MaxwellBoltzmannDistribution(atoms,temperature_K=temperature_K)
traj = Trajectory('out.traj', 'w', atoms)
# pfactor= 100 #120
pressure = 0
timestep = 0.2 * units.fs
dyn = MTTK(atoms,timestep=0.2*units.fs,run_steps=steps,t_stop=temperature_K,p_stop=pressure,pmode=None, tchain=3, pchain=3)
# dyn = NVTBerendsen(atoms, timestep=0.1*units.fs, temperature_K=300*units.kB, taut=0.5*1000*units.fs)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval=10)
dyn.attach(traj.write, interval=10)
dyn.run(steps)

"""

continue_pytemplate = """
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
from ase.md.nvtberendsen import NVTBerendsen
from mattersim.forcefield import MatterSimCalculator
calculator=MatterSimCalculator(device="cuda")
atoms = read("{structure}")
# 使用元素符号排序原子
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
steps = {steps}
temperature_K = {temperature}
traj = Trajectory('out.traj', 'w', atoms)
# pfactor= 100 #120
pressure = 0
timestep = 0.2 * units.fs
dyn = MTTK(atoms,timestep=0.2*units.fs,run_steps=steps,t_stop=temperature_K,p_stop=pressure,pmode=None, tchain=3, pchain=3)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval=10)
dyn.attach(traj.write, interval=10)
dyn.run(steps)

"""

nphugo_pytemplate = """
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
atoms = read("{structure}")
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
dyn = NPHugo(atoms, e0 = e0, p0 = p0, v0=v0, p_stop=pressure, timestep=timestep, tchain=3, pchain=3, pfreq=0.001)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval={dump_freq})
dyn.attach(traj.write, interval={dump_freq})
dyn.run(steps)
"""