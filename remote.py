#ase 读取 traj文件，在当前目录下创建名为task.{number}的文件夹，保存每一帧到其中
from ase.io import read,write
import shlex
import os
import shutil
from nepactive import dlog
import math
import paramiko
import subprocess
from typing import TYPE_CHECKING, Callable, Optional, Type, Union, Tuple, List, Any
import tarfile
import time
import socket
from glob import glob

import pathlib
import uuid
import json
from enum import IntEnum
from hashlib import sha1
import random
import numpy as np
# from dpdispatcher.contexts.ssh_context import SSHSession

def traj2tasks(traj_file:str,incar_file:str,potcar_file:str,frames:int=0, kpoint_file:Optional[str]=None):
    """
    需要在当前文件夹下准备好candidate.traj文件与INCAR，POTCAR文件,然后执行nepactive --remote
    """
    traj = read(traj_file, index=':')
    if frames == 0:
        frames = len(traj)
    index = random.sample(range(len(traj)),frames)
    index.sort()
    index = np.array(index,dtype="int32")
    sorted_traj = [traj[i] for i in index]
    np.savetxt("index.txt", index)
    for i, frame in enumerate(sorted_traj):
        folder_name = f'task.{i:06d}'
        os.makedirs(folder_name, exist_ok=True)
        write(f'{folder_name}/POSCAR', frame, format='vasp')
        if kpoint_file:
            shutil.copyfile(kpoint_file,f"{folder_name}/KPOINTS")
        shutil.copyfile(incar_file,f"{folder_name}/INCAR")
        shutil.copyfile(potcar_file,f"{folder_name}/POTCAR")
        #写好POTCAR和KPOINTS

class JobStatus(IntEnum):
    unsubmitted = 1
    waiting = 2
    running = 3
    terminated = 4
    finished = 5
    completing = 6
    unknown = 100

def rsync(
        from_file: str,
        to_file: str,
        port: int = 22,
        additional_args: Optional[List[str]] = None,
        key_filename: Optional[str] = None,
        timeout: Union[int, float] = 10,
    ):
        """Call rsync to transfer files.

        Parameters
        ----------
        from_file : str
            SRC
        to_file : str
            DEST
        port : int, default=22
            port for ssh
        key_filename : str, optional
            identity file name
        timeout : int, default=10
            timeout for ssh

        Raises
        ------
        RuntimeError
            when return code is not 0
        """
        ssh_cmd = [
            "ssh",
            "-o",
            "ConnectTimeout=" + str(timeout),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-p",
            str(port),
            "-q",
        ]
        if key_filename is not None:
            ssh_cmd.extend(["-i", key_filename])
        cmd = [
            "rsync",
            # -a: archieve
            # -z: compress
            "-az",
            "-e",
            " ".join(ssh_cmd),
            "-q",
            from_file,
            to_file
        ]
        if additional_args:
            cmd.extend(additional_args)
        ret, out, err = run_cmd_with_all_output(cmd, shell=False)
        if ret != 0:
            raise RuntimeError(f"Failed to run {cmd}: {err}")

def run_cmd_with_all_output(cmd, shell=True):
        with subprocess.Popen(
            cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as proc:
            out, err = proc.communicate()
            ret = proc.returncode
        return (ret, out, err)

class RetrySignal(Exception):
    """Exception to give a signal to retry the function."""

def retry(
    max_retry: int = 3,
    sleep: Union[int, float] = 60,
    catch_exception: Type[BaseException] = RetrySignal,
) -> Callable:
    """Retry the function until it succeeds or fails for certain times.

    Parameters
    ----------
    max_retry : int, default=3
        The maximum retry times. If None, it will retry forever.
    sleep : int or float, default=60
        The sleep time in seconds.
    catch_exception : Exception, default=Exception
        The exception to catch.

    Returns
    -------
    decorator: Callable
        The decorator.

    Examples
    --------
    >>> @retry(max_retry=3, sleep=60, catch_exception=RetrySignal)
    ... def func():
    ...     raise RetrySignal("Failed")
    """

    def decorator(func):
        assert max_retry > 0, "max_retry must be greater than 0"

        def wrapper(*args, **kwargs):
            current_retry = 0
            errors = []
            while max_retry is None or current_retry < max_retry:
                try:
                    return func(*args, **kwargs)
                except (catch_exception,) as e:
                    errors.append(e)
                    dlog.exception("Failed to run %s: %s", func.__name__, e)
                    # sleep certain seconds
                    dlog.warning("Sleep %s s and retry...", sleep)
                    time.sleep(sleep)
                    current_retry += 1
            else:
                # raise all exceptions
                raise RuntimeError(
                    "Failed to run %s for %d times" % (func.__name__, current_retry)
                ) from errors[-1]
        return wrapper
    return decorator

class Job:
    def __init__(self, idata:dict, group_tag:str, json_filepath, remote_subfilepath, task_list, stderr, status = JobStatus.unsubmitted, job_id = ""):
        """
        job_status: str, 'unsubmitted', 'submitted', 'finished', 'failed'
        stder, stdout : the error and output path of the job, not remote
        """
        self.idata = idata
        self.json_filepath = json_filepath
        self.remote_subfilepath = remote_subfilepath
        self.job_id = job_id
        self.task_list = task_list
        self.stderr = stderr
        self.status = status
        self.group_tag = group_tag
        self.fail_count = 0
        # self.serialize()
        # self.hash = self.get_hash()
        
        # self.stdout = stdout

    def get_hash(self):
        job_hash = sha1(json.dumps(self.cur_job).encode("utf-8")).hexdigest()
        # print(f"hash: {job_hash}")
        return job_hash

    def serialize(self):
        self.cur_job = {
            'idata': self.idata,
            'json_filepath': self.json_filepath,
            'remote_subfilepath': self.remote_subfilepath,
            'task_list': self.task_list,
            'job_id': self.job_id,
            # 'job_status': self.job_status,
            'stderr': self.stderr,
            'group_tag':self.group_tag
            # 'stdout': self.stdout,
        }
        # json.dump(cur_job, open(self.filepath, 'w')

    def dump(self,filename):
        self.serialize()
        with open(filename, 'w') as f:
            json.dump(self.cur_job, f)
            dlog.info(f"dump job to {filename}, job_id: {self.job_id}")

    @classmethod
    def deserialize(cls, json_file):
        with open(json_file,'r') as f:
            cur_job = json.load(f)
        return cls(idata=cur_job['idata'], group_tag=cur_job['group_tag'], json_filepath=cur_job['json_filepath'],remote_subfilepath=cur_job['remote_subfilepath'] , task_list=cur_job['task_list'], stderr = cur_job['stderr'], job_id = cur_job['job_id'])

    def set_job_id(self, job_id):
        self.job_id = job_id
    
    def set_job_status(self, status):
        self.status = status
    

command_script_template = """

cd {remote_root}
cd {task_work_path}
test $? -ne 0 && exit 1
if [ ! -f job_finished ] ;then
  {command} {log_err_part}
  if test $? -eq 0; then touch job_finished; else echo {task_work_path} >> ../failed_tasks;tail -v -c 1000 {err_file} > {last_err_file};fi
fi 

"""

end_script_template = """
cd {remote_root}
touch {group_tag}_job_tag_finished
"""

class Remotetask:
    def __init__(self,idata:dict):
        self.idata:dict = idata
        self.work_dir = os.getcwd()
        tar_suffix = idata.get('tar_suffix')
        self.project_name = idata.get('project_name')
        if tar_suffix is not None:
            self.tar_fileprefix = f"{self.project_name}_{tar_suffix}"
        else:
            self.tar_fileprefix = f"{self.project_name}"
        self.tar_filename = f"{self.tar_fileprefix}.tar.gz"
        self.recover:bool = idata.get('recover')

        # self.iter_num:int = iter_num
        self.ssh:paramiko.SSHClient = None
        # self.traj_filename = traj_filename
        self.look_for_keys = True
        self.key_filename = None
        self.username = idata.get('ssh_username')
        self.hostname = idata.get('ssh_hostname')
        self.remotename = f"{self.username}@{self.hostname}"
        self.port = idata.get('ssh_port')
        self.password = idata.get('ssh_password', None)

        self.remote_root = idata.get('remote_root')
        self.timeout = 10
        self.passphrase = None
        self.jobs:List[Job] = []
        self._setup_connection = False
        self.execute_command = idata.get('ssh_execute_command')
        self.totp_secret=None
        self.subfiles:list = []
        
        self._sftp = None
        
        self._setup_ssh()

        self.sftp = self.get_sftp()
    
    def glob_files(self,name="*"):
        path = os.path.join(self.work_dir, name)
        return glob(path)

    def setup(self):
        if not os.path.exists("task.000000"):
            traj_file = self.idata.get('remotetask_traj_file',"candidate.traj")
            incar_file = self.idata.get('incar_file',"INCAR")
            potcar_file = self.idata.get('potcar_file',"POTCAR")
            frames = self.idata.get('remotetask_frames',0)
            dlog.info(f"find traj_file:{traj_file}, incar_file:{incar_file}, potcar_file:{potcar_file}, frames:{frames}")
            # assert frames
            traj2tasks(traj_file=traj_file, incar_file=incar_file, potcar_file=potcar_file, frames=frames)
        self.fs = self.glob_files()
        self.tasks = self.glob_files("task.*")
        self.tasks = [os.path.basename(task) for task in self.tasks]

    def run_submission(self):
        # 创建SSH对象
        os.chdir(self.work_dir)
        # self.task_prepared = self.idata.get('task_prepared', False)
        self.setup()
        files = os.listdir(self.work_dir)
        cur_files = [file for file in files if file.startswith('cur_job')]
        jj = 3
        if cur_files:
            dlog.info(f"找到以下 'cur' 开头的文件:{cur_files}")
            jj =4
            self.jobs = [Job.deserialize(file) for file in cur_files]
        
        if os.path.exists("finished"):
            dlog.info("already finished")
            return

        if jj == 3:
            self.make_jobs()
            dlog.info("Tasks created")
            self.fs = self.glob_files()
            self.put_files()
            dlog.info("Files uploaded")
            jj = 4
        if jj == 4:
            dlog.info("Start to check status")
            for i,job in enumerate(self.jobs):
                job.status = self.check_status(job)
            finished = self.check_finished()
            while not finished:
                for i,job in enumerate(self.jobs):
                    job.status = self.check_status(job)
                    self.handle_job_status(job)
                # self.handle_job_status()
                finished = self.check_finished()
            # dlog.info("Jobs submitted")
            self.get_files()
            dlog.info("Files downloaded")
            os.chdir(self.work_dir)
            os.system("touch finished")
            clean = self.idata.get('clean', True)
            
            if clean:
                self.sftp.rmdir(f"{self.remote_root}/{self.tar_fileprefix}")
        else:
            raise ValueError("jj must be 3 or 4")
        
    def check_finished(self):
        statuses = [job.status for job in self.jobs]
        if all(status == JobStatus.finished for status in statuses):
            return True
        else:
            return False
        
    def handle_job_status(self,job:Job):
        job_state = job.status
        # self.fail_count = 0
        if job_state == JobStatus.unknown:
            raise RuntimeError(f"job_state for job {self} is unknown")

        if job_state == JobStatus.terminated:
            # job.fail_count += 1
            dlog.info(
                f"job: {job.remote_subfilepath} {job.job_id} terminated; "
                f"fail_count is {job.fail_count}; resubmitting job"
            )
            retry_count = 3
            if (job.fail_count) > 0 and (job.fail_count % retry_count == 0):
                last_error_message = self.get_last_error_message(job)
                err_msg = (
                    f"job:{job.remote_subfilepath} {job.job_id} failed {job.fail_count} times."
                )
                if last_error_message is not None:
                    err_msg += f"\nPossible remote error message: {last_error_message}"
                raise RuntimeError(err_msg)
            self.submit_job(job)
            if job.status != JobStatus.unsubmitted:
                dlog.info(
                    f"job:{job.remote_subfilepath} re-submit after terminated; new job_id is {job.job_id}"
                )
                time.sleep(0.2)
                job.status=self.check_status(job)
                dlog.info(
                    f"job:{job.remote_subfilepath} job_id:{job.job_id} after re-submitting; the state now is {repr(job.status)}"
                )

        if job_state == JobStatus.unsubmitted:
            dlog.debug(f"job: {job.group_tag} unsubmitted; submit it")
            # if self.fail_count > 3:
            #     raise RuntimeError("job:job {job} failed 3 times".format(job=self))
            self.submit_job(job=job)
            if job.status != JobStatus.unsubmitted:
                dlog.info(f"job: {job.remote_subfilepath} submit; job_id is {job.job_id}")

    def read_remote_file(self, fname:str):
        assert self.remote_root is not None
        self.ensure_alive()
        with self.sftp.open(
            pathlib.PurePath(os.path.join(self.remote_root, fname)).as_posix(),
            "r",
        ) as fp:
            ret = fp.read().decode("utf-8")
        return ret

    def get_last_error_message(self,job:Job) -> Optional[str]:
        """Get last error message when the job is terminated."""
        # assert self.machine is not None
        last_err_file = job.stderr
        if self.check_file_exists(last_err_file):
            last_error_message = self.read_remote_file(last_err_file)
            # red color
            last_error_message = "\033[31m" + last_error_message + "\033[0m"
            return last_error_message

    def check_file_exists(self, fname:str) -> bool:
        assert self.remote_root is not None
        self.ensure_alive()
        try:
            self.sftp.stat(
                pathlib.PurePath(os.path.join(f"{self.remote_root}/{self.tar_fileprefix}", fname)).as_posix()
            )
            ret = True
        except OSError:
            ret = False
        return ret

    def block_call(self, cmd):
        assert self.remote_root is not None
        self.ensure_alive()
        stdin, stdout, stderr = self.ssh.exec_command(
            (f"cd {shlex.quote(self.remote_root)} ;") + cmd
        )
        exit_status = stdout.channel.recv_exit_status()
        return exit_status, stdin, stdout, stderr
    
    def block_checkcall(self, cmd, asynchronously=False) -> Tuple[Any, Any, Any]:
        """Run command with arguments. Wait for command to complete.

        Parameters
        ----------
        cmd : str
            The command to run.
        asynchronously : bool, optional, default=False
            Run command asynchronously. If True, `nohup` will be used to run the command.

        Returns
        -------
        stdin
            standard inout
        stdout
            standard output
        stderr
            standard error

        Raises
        ------
        RuntimeError
            when the return code is not zero
        """
        if asynchronously:
            cmd = f"nohup {cmd} >/dev/null &"
        exit_status, stdin, stdout, stderr = self.block_call(cmd)
        if exit_status != 0:
            raise RuntimeError(
                "Get error code %d in calling %s. message: %s"
                % (
                    exit_status,
                    cmd,
                    stderr.read().decode("utf-8"),
                )
            )
        return stdin, stdout, stderr
    
    def put_files(self):
        self.tar_files()
        try:
            self.sftp.mkdir(f"{self.remote_root}/{self.tar_fileprefix}")
        except OSError:
            pass

        self.rsync(from_f = self.tar_filename, to_f = f"{self.idata.get('remote_root')}/{self.tar_fileprefix}")

        self.block_checkcall(f"cd {self.remote_root}/{self.tar_fileprefix} &&tar -xzf {self.tar_filename}")

        os.remove(self.tar_filename)

    def check_status(self, job:Job):
        job_id = job.job_id
        dlog.debug(f"checking job: {job_id}")
        # raise NotImplementedError
        if job_id == "":
            return JobStatus.unsubmitted
        command = 'squeue -h -o "%.18i %.2t" -j ' + job_id
        # print(job_id)
        ret, stdin, stdout, stderr = self.block_call(command)
        # print(f"stderr: {stderr.read().decode('utf-8')},stdout:{stdout.read().decode('utf-8')}")
        #只能read一次
        dlog.debug(f"ret: {ret}")
        # print(f"ret: {ret}")
        if ret != 0:
            err_str = stderr.read().decode("utf-8")
            print(f"err_str: {err_str}")
            if "Invalid job id specified" in err_str:
                if self.check_finish_tag(job):
                    dlog.info(f"job: {job.group_tag} {job.job_id} finished")
                    return JobStatus.finished
                else:
                    return JobStatus.terminated
            elif (
                "Socket timed out on send/recv operation" in err_str
                or "Unable to contact slurm controller" in err_str
            ):
                # retry 3 times
                raise RetrySignal(
                    "Get error code %d in checking status with job: %s . message: %s"
                    % (ret, job.group_tag, err_str)
                )
            raise RuntimeError(
                "status command (%s) fails to execute.\n"
                "job_id:%s \n error message:%s\n return code %d\n"
                % (command, job_id, err_str, ret)
            )
        status_lines = stdout.read().decode("utf-8").split("\n")[:-1]
        status = []
        for status_line in status_lines:
            status_word = status_line.split()[-1]
            if not (len(status_line.split()) == 2 and status_word.isupper()):
                raise RuntimeError(
                    "Error in getting job status, "
                    + f"status_line = {status_line}, "
                    + f"parsed status_word = {status_word}"
                )
            if status_word in ["PD", "CF", "S"]:
                status.append(JobStatus.waiting)
            elif status_word in ["R"]:
                status.append(JobStatus.running)
            elif status_word in ["CG"]:
                status.append(JobStatus.completing)
            elif status_word in [
                "C",
                "E",
                "K",
                "BF",
                "CA",
                "CD",
                "F",
                "NF",
                "PR",
                "SE",
                "ST",
                "TO",
            ]:
                status.append(JobStatus.finished)
            else:
                status.append(JobStatus.unknown)
        # running if any job is running
        if JobStatus.running in status:
            return JobStatus.running
        elif JobStatus.waiting in status:
            return JobStatus.waiting
        elif JobStatus.completing in status:
            return JobStatus.completing
        elif JobStatus.unknown in status:
            return JobStatus.unknown
        else:
            if self.check_finish_tag(job):
                dlog.info(f"job: {job.group_tag} {job.job_id} finished")
                return JobStatus.finished
            else:
                return JobStatus.terminated
    
    def check_finish_tag(self, job:Job):
        job_tag_finished = f"{job.group_tag}_job_tag_finished"  # 假设这是文件路径
        print(f"job_tag_finished: {job_tag_finished}")
        if self.check_file_exists(job_tag_finished):
            return True
        else:
            return False
        

    @retry(max_retry=3, sleep=60, catch_exception=RetrySignal)
    def submit_job(self,job:Job):
        # command = f"cd {self.remote_root} && sbatch {shlex.quote(job.remote_subfilepath)}"
        # self.block_checkcall(command)

        job.fail_count += 1
        print(f"group_tag: {job.group_tag}")
        command = "cd {} && {} {}".format(
        shlex.quote(f"{self.remote_root}/{self.tar_fileprefix}"),
        "sbatch",
        shlex.quote(f"{job.group_tag}.sub"),
        )
        ret, stdin, stdout, stderr = self.block_call(command)
        if ret != 0:
            err_str = stderr.read().decode("utf-8")
            if (
                "Socket timed out on send/recv operation" in err_str
                or "Unable to contact slurm controller" in err_str
            ):
                # server network error, retry 3 times
                raise RetrySignal(
                    "Get error code %d in submitting with job: %s . message: %s"
                    % (ret, job.group_tag, err_str)
                )
            elif (
                "Job violates accounting/QOS policy" in err_str
                # the number of jobs exceeds DEFAULT_MAX_JOB_COUNT (by default 10000)
                or "Slurm temporarily unable to accept job, sleeping and retrying"
                in err_str
            ):
                # job number exceeds, skip the submitting
                return ""
            raise RuntimeError(
                "command %s fails to execute\nerror message:%s\nreturn code %d\n"
                % (command, err_str, ret)
            )
        subret:str = stdout.readlines()
        # print(f"subret:{subret}")
        job_id = subret[0].split(" ")[-1].strip()
        # print(f"job_id:{job_id}")
        # exit()
        dlog.info(f"job:{job.group_tag} submitted as :{job_id}")
        job.set_job_id(job_id)
        job.status = self.check_status(job)
        job.dump(f"{self.work_dir}/cur_job_{job.group_tag}.json")
        
    def get_files(self):
        # self.remote_tar_files()
        dlog.info(f"beginning to rsync files in {self.work_dir}")
        #/号很重要
        additional_args = ["--exclude=*.tar.gz", "--exclude=*log", "--exclude=*out","--exclude=*/*.xml", "--exclude=*/POTCAR","--exclude=*/XDATCAR"]
        self.rsync(from_f = f"{self.remote_root}/{self.tar_fileprefix}/", to_f = self.work_dir, send = False, additional_args=additional_args)
        dlog.info(f"rsync files in {self.work_dir} successfully")
        # with tarfile.open(self.tar_filename, mode='r:gz') as tar:
            # tar.extractall(path=self.work_dir)

    def remote_tar_files(self):

        # ntar = len(self.fs)
        tar_command = "czfh"
        of = self.tar_filename
        dir = f"{self.remote_root}/{self.tar_fileprefix}"
        # file_list = " ".join([shlex.quote(file) for file in self.fs])
        # 包含主文件夹
        # tar_cmd = f"cd {self.remote_root} && tar {tar_command} {shlex.quote(of)} {self.tar_fileprefix}"
        # self.block_checkcall(tar_cmd)
        dlog.info(f"beginning to tar files in {dir}")
        per_nfile = 100
        ntar = len(self.fs) // per_nfile + 1
        # if ntar <= 1:
        #     file_list = " ".join([shlex.quote(os.path.basename(file)) for file in self.tasks])
        #     tar_cmd = f"cd {dir} && tar {tar_command} {shlex.quote(of)} {file_list}"
        # else:
        #     file_list_file = pathlib.PurePath(
        #         os.path.join(f"{self.remote_root}/{self.tar_fileprefix}", f".tmp_tar_{uuid.uuid4()}")
        #     ).as_posix()
        #     self.write_file(file_list_file, "\n".join(self.tasks))
        #     tar_cmd = (
        #         f"cd {dir} &&tar {tar_command} {shlex.quote(of)} -T {shlex.quote(os.path.basename(file_list_file))}"
        #     )
        tar_cmd = f"cd {dir} && tar {tar_command} {shlex.quote(of)} *"
        try:
            self.block_checkcall(tar_cmd)
        except RuntimeError as e:
            if "No such file or directory" in str(e):
                raise FileNotFoundError(
                    f"Backward files do not exist in the remote directory in executing command {tar_cmd}"
                ) from e
            raise e
        
        self.block_checkcall(f"rm -r {self.remote_root}/{self.tar_fileprefix}")

    def write_file(self, fname, write_str):
        assert self.remote_root is not None
        self.ensure_alive()
        dlog.info(f"tar_file_prefix: {self.tar_fileprefix}")
        fname = pathlib.PurePath(fname).as_posix()
        # to prevent old file from being overwritten but cancelled, create a temporary file first
        # when it is fully written, rename it to the original file name
        temp_fname = fname + "_tmp"
        try:
            with self.sftp.open(temp_fname, "w") as fp:
                fp.write(write_str)
            # Rename the temporary file
            self.block_checkcall(f"mv {shlex.quote(temp_fname)} {shlex.quote(fname)}")
            dlog.info(f"Successfully wrote to file {fname}")
        # sftp.rename may throw OSError
        except OSError as e:
            dlog.exception(f"Error writing to file {fname}")
            raise e

    def rsync(self, from_f, to_f, send=True, additional_args=None):
        # from_f = None
        # to_f = None
        # ssh = paramiko.SSHClient()
        try:
            if self.sftp is None:
                raise RuntimeError("SFTP connection is not established.")
            self.sftp.mkdir(f"{self.remote_root}/{self.tar_fileprefix}")
        except OSError:
                pass
        # os.chdir(self.remote_root)
        key_filename = self.key_filename
        # ssh.connect(hostname = self.hostname, port = self.port, username = self.username, password = self.password)
        timeout = self.idata.get('ssh_timeout')

        assert os.path.abspath(to_f)
        
        ####注意是否按照顺序来传输文件
        if send is True:
            to_f = self.remotename + ":" + to_f
        else:
            from_f = self.remotename + ":" + from_f

        rsync(
                    from_f,
                    to_f,
                    port=self.port,
                    key_filename=key_filename,
                    timeout=timeout,
                    additional_args=additional_args
                )

    def tar_files(self):
        of = self.tar_filename
        tarfile_mode = "w:gz"
        dereference=True
        directories=None
        kwargs = {"compresslevel": 6}
        files = self.fs
        # print(f"tar files {files} to {of}")
        if os.path.isfile(os.path.join(self.work_dir, of)):
            os.remove(os.path.join(self.work_dir, of))
        with tarfile.open(
            os.path.join(self.work_dir, of),
            tarfile_mode,
            **kwargs,
        ) as tar:
            # avoid compressing duplicated files or directories
            for ii_full in set(files):
                ii = os.path.basename(ii_full)
                # print(f"adding {ii_full} as {ii}")
                tar.add(ii_full, arcname=ii)
            if directories is not None:
                for ii_full in set(directories):
                    ii = os.path.basename(ii_full)
                    tar.add(ii_full, arcname=ii, recursive=False)
            # self.ensure_alive()
        dlog.info(f"tar file {of} created")

    def _setup_ssh(self):
        # machine = self.machine
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self.key_filename = self.idata.get('ssh_key_filename')
        hostname = self.idata.get('ssh_hostname')
        port = self.idata.get('ssh_port')
        username = self.idata.get('ssh_username')
        print(f"hostname: {hostname}, port: {port}, username: {username}")
        self.ssh.connect(
                    hostname=hostname,
                    port=port,
                    username=username,
                    # password=password,
                    # key_filename = key_filename,
                    timeout=30,
                    compress=True,
                    allow_agent=False, 
                    look_for_keys=True
                )
        dlog.info("ssh connection established")
        command = f"""
                REMOTE_ROOT = {self.remote_root}
                cd $REMOTE_ROOT
                    """
        self.ssh.exec_command(command)
        if self.execute_command is not None:
            self.ssh.exec_command(self.execute_command)
        self._setup_connection == True

    def get_sftp(self):
        if self._sftp is None:
            assert self.ssh is not None
            self.ensure_alive()
            self._sftp = self.ssh.open_sftp()
        return self._sftp      #在内层不会导致无返回报错吗
        
    def ensure_alive(self, max_check=10, sleep_time=10):
        count = 1
        while not self._check_alive():
            if count == max_check:
                raise RuntimeError(
                    "cannot connect ssh after %d failures at interval %d s"
                    % (max_check, sleep_time)
                )
            dlog.info("connection check failed, try to reconnect to " + self.hostname)
            self._setup_ssh()
            count += 1
            time.sleep(sleep_time)

    def _check_alive(self):
        if self.ssh is None:
            return False
        try:
            transport = self.ssh.get_transport()
            assert transport is not None
            transport.send_ignore()
            return True
        except EOFError:
            return False

    def make_jobs(self):
        groups = self.group_tasks()
        for i,group in enumerate(groups):
            # print(f"grouptype:{type(group)}")
            assert isinstance(group,list)
            header_script:list = self.idata.get('slurm_header_script')
            header_script = "\n".join(header_script)
            cycle_run_script = ""
            for ii,task_work_path in enumerate(group):
                log_err_part = f"1>>fp.log 2>>fp.log"
                cycle_run_script += command_script_template.format(
                                                                    remote_root = f"{self.remote_root}/{self.tar_fileprefix}",
                                                                    task_work_path = task_work_path,
                                                                    log_err_part = log_err_part,
                                                                    err_file = 'fp.log',
                                                                    last_err_file = f'group_{i:03d}_last_err',
                                                                    command = self.idata.get('fp_command')
                                                                )
            subfile = f"group_{i:03d}.sub"
            self.subfiles.extend(subfile)
            end_script = end_script_template.format(remote_root = f"{self.remote_root}/{self.tar_fileprefix}",
                                                    group_tag = f"group_{i:03d}")
            total_script = "\n".join([header_script, cycle_run_script, end_script])
            header_script + cycle_run_script
            json_filepath = f"{self.work_dir}/cur_job_{ii}.json.json"
            remote_subfilepath = f"{self.remote_root}/{self.tar_fileprefix}/{subfile}"
            stderr = f'{self.remote_root}/{self.tar_fileprefix}/group_{i:03d}_last_err'
            job = Job(idata = self.idata, json_filepath = json_filepath,stderr = stderr,
                       remote_subfilepath = remote_subfilepath, task_list = groups[i], group_tag = f"group_{i:03d}")
            self.jobs.append(job)
            with open(subfile, 'w') as f:
                f.write(total_script)
                
    def group_tasks(self):
        group_numbers = self.idata.get('group_numbers',4)
        dlog.info(f"will split tasks into {group_numbers} groups")
        task_numbers = len(self.tasks)
        group_size = math.ceil(task_numbers/ group_numbers)
        groups = [self.tasks[i:i+group_size] for i in range(0, task_numbers, group_size)]
        return groups
