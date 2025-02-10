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
# from dpdispatcher.contexts.ssh_context import SSHSession

def rsync(
        from_file: str,
        to_file: str,
        port: int = 22,
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
            to_file,
        ]
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
    def __init__(self, idata:dict, filepath, remote_filepath, task_list):
        self.idata = idata
        self.filepath = filepath
        self.remote_filepath = remote_filepath
        self.job_id = None
        self.task_list = task_list

    def set_job_id(self, job_id):
        self.job_id = job_id
    


command_script_template = """

cd $REMOTE_ROOT
cd {task_work_path}
test $? -ne 0 && exit 1
if [ ! -f job_finished ] ;then
  {command} {log_err_part}
  if test $? -eq 0; then touch job_finished; else echo {task_work_path} >> $REMOTE_ROOT/failed_tasks;tail -v -c 1000 $REMOTE_ROOT/{task_work_path}/{err_file} > $REMOTE_ROOT/{last_err_file};fi
fi &

"""

class Remote_profile:
    def __init__(self, idata):
        self.username = idata.get('ssh_username')
        self.hostname = idata.get('ssh_hostname')
        self.remotename = f"{self.username}@{self.hostname}"
        port = idata.get('ssh_port')



class Remotetask:
    def __init__(self, work_dir, iter_num, idata:dict, traj_filename):
        self.idata:dict = idata
        self.work_dir = idata.get('work_dir') #总的工作目录
        self.fp_dir = idata.get('fp_dir')
        self.label_dir = f"{self.work_dir}/iter.{iter_num:06d}/02.label"
        self.make_tasks(self.laber_dir, idata)
        #传输任务提交任务
        self.remote = Remote_profile(idata)
        project_name = idata.get('project_name')
        self.tar_fileprefix = f"{project_name}_{iter_num:06d}"
        self.tar_filename = f"{self.tar_fileprefix}.tar.gz"
        self.recover:bool = idata.get('recover')
        self.work_dir = work_dir
        self.iter_num:int = iter_num
        self.ssh = None
        self.local_root = os.path.basename(self.laber_dir)
        self.traj_filename = traj_filename
        self.look_for_keys = True
        self.key_filename = None
        self.username = idata.get('ssh_username')
        self.hostname = idata.get('ssh_hostname')
        self.remotename = f"{self.username}@{self.hostname}"
        self.port = idata.get('ssh_port')
        self.password = idata.get('ssh_password', None)
        self.fs = self.glob_files()
        self.remote_root = idata.get('remote_root')
        self.timeout = 10
        self.passphrase = None
        self.jobs:List[Job]


        self._setup_ssh()
    
    def glob_files(self):
        return glob(f"{self.laber_dir}/*")

    def run_submission(self):
        # 创建SSH对象
        self.make_tasks()
        dlog.info("Tasks created")
        self.put_files()
        dlog.info("Files uploaded")
        self.sub_jobs()
        dlog.info("Jobs submitted")
        self.get_files()
        dlog.info("Files downloaded")

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
                "Get error code %d in calling %s with job: %s . message: %s"
                % (
                    exit_status,
                    cmd,
                    self.iter_num,
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

        self.block_checkcall(f"tar -xzf {self.tar_filename}")

    def sub_jobs(self):
        job_status = self.check_job_status()
        remote_root = os.path.join(self.remote_root,self.tar_fileprefix)
        if os.path.isfile(os.path.join(self.label_dir, "job_id")):
            with open(os.path.join(self.label_dir, "job_id")) as f:
                self.job_ids = f.readlines()
        else:
            self.job_ids = []
        for ii in self.sub_jobs:
            command = "cd {} && {} {}".format(
            shlex.quote(remote_root),
            "sbatch",
            shlex.quote(ii),
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
                        % (ret, job.job_hash, err_str)
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
            subret = stdout.readlines()
            # --parsable
            # Outputs only the job id number and the cluster name if present.
            # The values are separated by a semicolon. Errors will still be displayed.
            job_id = subret[0].split(";")[0].strip()
            self.job_ids.extend(job_id)
        
        with open(os.path.join(self.label_dir, "job_id"), "w") as f:    
            f.writelines(self.job_ids+"\n")
        

    def get_job_status(self):
        command = "squeue -u {} -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R'".format(self.idata.get('user'))
        stdin, stdout, stderr = self.block_checkcall(command)
        return stdout.read().decode("utf-8")

    def get_files(self):
        self.remote_tar_files()
        iter_dir = os.path.join(self.work_dir, f"iter.{self.iter_num:06d}")
        os.chdir(iter_dir)
        self.rsync(from_f = self.remote_root, to_f = iter_dir, send = False)
        with tarfile.open(self.tar_filename, mode='r:gz') as tar:
            tar.extractall(path=iter_dir)

    def remote_tar_files(self):
        # task_dir = os.path.join(self.remote_root, self.tar_fileprefix)
        # ntar = len(self.fs)
        tar_command = "czfh"
        of = self.tar_filename
        dir = f"{self.remote_root}/{self.tar_fileprefix}"
        # file_list = " ".join([shlex.quote(file) for file in self.fs])
        # 包含主文件夹
        # tar_cmd = f"cd {self.remote_root} && tar {tar_command} {shlex.quote(of)} {self.tar_fileprefix}"
        # self.block_checkcall(tar_cmd)
        per_nfile = 100
        ntar = len(self.fs) // per_nfile + 1
        if ntar <= 1:
            file_list = " ".join([shlex.quote(file) for file in self.fs])
            tar_cmd = f"tar {tar_command} {shlex.quote(of)} {file_list}"
        else:
            file_list_file = pathlib.PurePath(
                os.path.join(self.remote_root, f".tmp_tar_{uuid.uuid4()}")
            ).as_posix()
            self.write_file(file_list_file, "\n".join(self.fs))
            tar_cmd = (
                f"cd {self.remote_root}/{self.tar_fileprefix} &&tar {tar_command} {shlex.quote(of)} -T {shlex.quote(file_list_file)}"
            )
        try:
            self.block_checkcall(tar_cmd)
        except RuntimeError as e:
            if "No such file or directory" in str(e):
                raise FileNotFoundError(
                    "Backward files do not exist in the remote directory."
                ) from e
            raise e
        
        self.sftp.remove(dir)

    def write_file(self, fname, write_str):
        assert self.remote_root is not None
        self.ensure_alive()
        fname = pathlib.PurePath(os.path.join(self.remote_root, fname)).as_posix()
        # to prevent old file from being overwritten but cancelled, create a temporary file first
        # when it is fully written, rename it to the original file name
        temp_fname = fname + "_tmp"
        try:
            with self.sftp.open(temp_fname, "w") as fp:
                fp.write(write_str)
            # Rename the temporary file
            self.block_checkcall(f"mv {shlex.quote(temp_fname)} {shlex.quote(fname)}")
        # sftp.rename may throw OSError
        except OSError as e:
            dlog.exception(f"Error writing to file {fname}")
            raise e

    def rsync(self, from_f, to_f, send=True):
        # from_f = None
        # to_f = None
        # ssh = paramiko.SSHClient()
        try:
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
                )

    def tar_files(self):
        of = self.tar_filename
        tarfile_mode = "w:gz"
        dereference=True
        directories=None
        kwargs = {"compresslevel": 6}
        files = self.fs
        if os.path.isfile(os.path.join(self.label_dir, of)):
            os.remove(os.path.join(self.label_dir, of))
            with tarfile.open(
                os.path.join(self.label_dir, of),
                tarfile_mode,
                dereference=dereference,
                **kwargs,
            ) as tar:
                # avoid compressing duplicated files or directories
                for ii_full in set(files):
                    ii = os.path.basename(ii_full)
                    tar.add(ii_full, arcname=ii)
                if directories is not None:
                    for ii_full in set(directories):
                        ii = os.path.basename(ii_full)
                        tar.add(ii_full, arcname=ii, recursive=False)
            self.ensure_alive()

    def _setup_ssh(self):
        # machine = self.machine
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        # if self.totp_secret and self.password is None:
        #     self.password = generate_totp(self.totp_secret)
        # self.ssh.connect(hostname=self.hostname, port=self.port,
        #                 username=self.username, password=self.password,
        #                 key_filename=self.key_filename, timeout=self.timeout,passphrase=self.passphrase,
        #                 compress=True,
        #                 )
        # assert(self.ssh.get_transport().is_active())
        # transport = self.ssh.get_transport()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.hostname, self.port))

        # Make a Paramiko Transport object using the socket
        ts = paramiko.Transport(sock)
        ts.banner_timeout = 60
        ts.auth_timeout = self.timeout + 20
        ts.use_compression(compress=True)

        # Tell Paramiko that the Transport is going to be used as a client
        ts.start_client(timeout=self.timeout)

        # Begin authentication; note that the username and callback are passed
        key = None
        key_ok = False
        key_error = None
        keyfiles = []
        if self.key_filename:
            key_path = os.path.abspath(self.key_filename)
            if os.path.exists(key_path):
                for pkey_class in (
                    paramiko.RSAKey,
                    paramiko.DSSKey,
                    paramiko.ECDSAKey,
                    paramiko.Ed25519Key,
                ):
                    try:
                        # passing empty passphrase would not raise error.
                        key = pkey_class.from_private_key_file(
                            key_path, self.passphrase
                        )
                    except paramiko.SSHException as e:
                        pass
                    if key is not None:
                        break
            else:
                raise OSError(f"{key_path} not found!")
        elif self.look_for_keys:
            for keytype, name in [
                (paramiko.RSAKey, "rsa"),
                (paramiko.DSSKey, "dsa"),
                (paramiko.ECDSAKey, "ecdsa"),
                (paramiko.Ed25519Key, "ed25519"),
            ]:
                for directory in [".ssh", "ssh"]:
                    full_path = os.path.join(
                        os.path.expanduser("~"), directory, f"id_{name}"
                    )
                    if os.path.isfile(full_path):
                        keyfiles.append((keytype, full_path))
                        # TODO: supporting cert
            for pkey_class, filename in keyfiles:
                try:
                    key = pkey_class.from_private_key_file(filename, self.passphrase)
                    self.key_filename = filename 
                except paramiko.SSHException as e:
                    pass
                if key is not None:
                    break

        allowed_types = set()
        if key is not None:
            try:
                allowed_types = set(ts.auth_publickey(self.username, key))
            except paramiko.ssh_exception.AuthenticationException as e:
                key_error = e
            else:
                key_ok = True
        if self.totp_secret is not None or "keyboard-interactive" in allowed_types:
            try:
                ts.auth_interactive(self.username, self.inter_handler)
            except paramiko.ssh_exception.AuthenticationException:
                # since the asynchrony of interactive authentication, one addtional try is added
                # retry for up to 6 times
                raise RetrySignal("Authentication failed")
            self._keyboard_interactive_auth = True
        elif key_ok:
            pass
        elif self.password is not None:
            ts.auth_password(self.username, self.password)
        elif key_error is not None:
            raise RuntimeError(
                "Authentication failed, try to provide password"
            ) from key_error
        else:
            raise RuntimeError("Please provide at least one form of authentication")
        assert ts.is_active()
        # Opening a session creates a channel along the socket to the server
        try:
            ts.open_session(timeout=self.timeout)
        except paramiko.ssh_exception.SSHException:
            # retry for up to 6 times
            # ref: https://github.com/paramiko/paramiko/issues/1508
            raise RetrySignal("Opening session failed")
        ts.set_keepalive(60)
        self.ssh._transport = ts  # type: ignore
        # reset sftp
        self._sftp = None
        if self.execute_command is not None:
            self.exec_command(self.execute_command)

    @property
    def sftp(self):
        if self._sftp is None:
            assert self.ssh is not None
            self.ensure_alive()
            self._sftp = self.ssh.open_sftp()
            return self._sftp
        
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
        os.chdir(self.label_dir)
        file_name = self.traj_filename
        traj = read(file_name, index=':')
        self.subfiles = []
        if not (self.idata.get('incar_file') or self.idata.get('kpoints_file') or self.idata.get('potcar_file')):
                raise ValueError('incar_file, kpoints_file, potcar_file must be provided')
        shutil.copyfile(self.idata["incar_file"], "INCAR")
        tasks:list
        # shutil.copyfile(idata["kpoints_file"], "KPOINTS")
        # shutil.copyfile(idata["potcar_file"], "POTCAR")
        for i, frame in enumerate(traj):
            folder_name = f'task.{i:06d}'
            os.makedirs(folder_name, exist_ok=True)
            tasks.extend(folder_name)
            write(f'{folder_name}/POSCAR', frame, format='vasp')
            #写好POTCAR和KPOINTS
        groups = self.group_tasks()
        
        for i,group in enumerate(groups):
            header_script = self.idata.get('header_script')
            cycle_run_script = ""
            for ii,task_work_path in group:

                log_err_part = f"1>>fp.log 2>>fp.log"
                cycle_run_script += command_script_template.format(
                                                                    task_work_path = task_work_path,
                                                                    log_err_part = log_err_part,
                                                                    err_file = 'fp.log',
                                                                    last_err_file = f'group_{i:03d}_last_err',
                                                                    command = self.idata.get('fp_command')
                                                                )
            subfile = f"group_{i:03d}.sub"
            self.subfiles.extend(subfile)
            total_script = "\n".join([header_script, cycle_run_script])
            header_script + cycle_run_script
            job = Job(idata = self.idata, filepath = f"{self.label_dir}/{subfile}",
                       remote_filepath = self.remote_root, task_list = groups[i])
            self.jobs.append(job)
            with open(subfile) as f:
                f.write(total_script)
                
    def group_tasks(tasks:list, idata:dict):
        group_numbers = idata.get('group_numb',4)
        task_numbers = len(tasks)
        group_size = math.ceil(task_numbers/ group_numbers)
        groups = [tasks[i:i+group_size] for i in range(0, task_numbers, group_size)]
        return groups
