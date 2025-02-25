
msst_template = """
potential		nep.txt

minimize sd 1.0e-6 1000

time_step	    {time_step}
velocity		300

ensemble        nvt_ber 300 300 200 # 0 0 0 10 0 0 258 258 258 77.5 77.5 253 1000

# dump_thermo		{dump_freq}

run			    20000

dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
ensemble msst x {v_shock} qmass {qmass} mu {viscosity} tscale 0.01
run                         {run_steps}
"""

nvt_template = """
potential		nep.txt
minimize sd 1.0e-6 1000

time_step	    {time_step}
velocity		{temperature}

ensemble        nvt_ber {temperature} {temperature} 200 # 0 0 0 10 0 0 258 258 258 77.5 77.5 253 1000

dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
run			    {run_steps}
"""

npt_template = """
potential		nep.txt
minimize sd 1.0e-6 10000

time_step	    {time_step}
velocity		{temperature}

ensemble        npt_ber {temperature} {temperature} 200 {pressure} {pressure} {pressure} 50 50 50 1000

dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
run			    {run_steps}
"""

nphugo_template = """
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
ensemble nphug iso {pressure} {pressure} e0 {e0} p0 {p0} v0 {v0}

run                       {run_steps}
"""

model_devi_template = """
cd {work_dir}
cd {task_dir}
rm -f dump.xyz thermo.out log
# if [ ! -f task_finished ] ;then 
gpumd > log 2>&1
# if test $? -eq 0; then touch task_finished;fi
# fi

"""

nep_in_template ="""
type 4 H C N O
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
generation    10000  # default
 
"""