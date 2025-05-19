import os
import csv
import json
import yaml
import shutil
import logging
from ase.io import ParseError
from ase.io.vasp import read_vasp_out
from dpdispatcher import Machine, Resources, Task, Submission
from ion_CSP.log_and_time import redirect_dpdisp_logging
from ion_CSP.identify_molecules import identify_molecules, molecules_information


class VaspProcessing:
    def __init__(self, work_dir: str):
        redirect_dpdisp_logging(os.path.join(work_dir, "dpdispatcher.log"))
        self.base_dir = work_dir
        os.chdir(self.base_dir)
        self.for_vasp_opt_dir = f"{work_dir}/3_for_vasp_opt"
        self.vasp_optimized_dir = f"{work_dir}/4_vasp_optimized"
        self.param_dir = os.path.join(os.path.dirname(__file__), "../../param")

    def dpdisp_vasp_tasks(
        self,
        machine: str,
        resources: str,
        nodes: int = 1,
    ):
        """
        Based on the dpdispatcher module, prepare and submit files for optimization on remote server or local machine.
        """
        # 调整工作目录，减少错误发生
        os.chdir(self.for_vasp_opt_dir)
        # 读取machine.json和resources.json的参数
        if machine.endswith(".json"):
            machine = Machine.load_from_json(machine)
        elif machine.endswith(".yaml"):
            machine = Machine.load_from_yaml(machine)
        else:
            raise KeyError("Not supported machine file type")
        if resources.endswith(".json"):
            resources = Resources.load_from_json(resources)
        elif resources.endswith(".yaml"):
            resources = Resources.load_from_yaml(resources)
        else:
            raise KeyError("Not supported resources file type")
        # 由于dpdispatcher对于远程服务器以及本地运行的forward_common_files的默认存放位置不同，因此需要预先进行判断，从而不改动优化脚本
        machine_inform = machine.serialize()
        if machine_inform["context_type"] == "SSHContext":
            # 如果调用远程服务器，则创建二级目录
            parent = "data/"
        elif machine_inform["context_type"] == "LocalContext":
            # 如果在本地运行作业，则只在后续创建一级目录
            parent = ""

