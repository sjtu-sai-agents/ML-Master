"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
- parallel execution
"""

import logging
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue, Lock
from pathlib import Path

import humanize
from dataclasses_json import DataClassJsonMixin

from types import SimpleNamespace

def trim_long_string(string, threshold=5100, k=2500):
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string

logger = logging.getLogger("ml-master")

@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None



def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
    if format_tb_ipython:
        import IPython.core.ultratb

        # tb_offset = 1 to skip parts of the stack trace in weflow code
        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [l for l in tb_lines if "ml-master/" not in l and "importlib" not in l]
        )
        # tb_str = "".join([l for l in tb_lines])

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg:str):
        self.queue.put(msg)

    def flush(self):
        pass


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        max_parallel_run: int = 3,
        cfg = None
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        
        # self.process: Process = None  # type: ignore
        self.max_parallel_run = cfg.agent.search.parallel_search_num if cfg.agent.search.parallel_search_num else max_parallel_run
        self.agent_file_name = [f"runfile_{i}.py" for i in range(self.max_parallel_run)]
        self.process: list[Process] = [None] * self.max_parallel_run
        self.current_parallel_run = 0 #
        self.status_map = [0] * self.max_parallel_run
        self.code_inq = [None] * self.max_parallel_run
        self.result_outq = [None] * self.max_parallel_run
        self.event_outq = [None] * self.max_parallel_run
        self.start_cpu_id = int(cfg.start_cpu_id)
        self.cpu_number = int(cfg.cpu_number)
        if self.cpu_number < self.max_parallel_run:
            raise ValueError("The maximum level of parallelism exceeds the number of allocated CPU cores; ensure that each process has at least one CPU core.")
        self.lock = Lock()

    def check_current_status(self):
        '''
        check current parallel run number
        '''
        if self.current_parallel_run < self.max_parallel_run:
            # logger.info(f"current_parallel_run:{self.current_parallel_run},max parallel run:{self.max_parallel_run}")
            # logger.info(f"check_current_status return True")

            return True
        else:
            # logger.info(f"current_parallel_run:{self.current_parallel_run},max parallel run:{self.max_parallel_run}")
            # logger.info(f"check_current_status return false")

            return False

    # def child_proc_setup(self, result_outq: Queue) -> None:
    #     # disable all warnings (before importing anything)
    #     import shutup

    #     shutup.mute_warnings()
    #     os.chdir(str(self.working_dir))

    #     # this seems to only  benecessary because we're exec'ing code from a string,
    #     # a .py file should be able to import modules from the cwd anyway
    #     sys.path.append(str(self.working_dir))

    #     # capture stdout and stderr
    #     # trunk-ignore(mypy/assignment)
    #     sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def child_proc_setup(self, result_outq: Queue) -> None:
        try:
            print("[child_proc_setup] import shutup...")
            import shutup

            print("[child_proc_setup] mute_warnings...")
            shutup.mute_warnings()

            print(f"[child_proc_setup] chdir to {self.working_dir}...")
            os.chdir(str(self.working_dir))

            print("[child_proc_setup] append to sys.path...")
            sys.path.append(str(self.working_dir))

            print("[child_proc_setup] redirect stdout/stderr...")
            sys.stdout = sys.stderr = RedirectQueue(result_outq)

        except Exception as e:
            import traceback
            result_outq.put("[child_proc_setup error] " + traceback.format_exc())
            raise


    def replace_submission_name(self, code, _id):
        submission_file_name = f"submission_{_id}.csv"
        modified_code = code
        if "submission/submission.csv" in code:
            modified_code = code.replace("submission/submission.csv", f"submission/{submission_file_name}")
        if "/submission.csv" in modified_code:
            modified_code = modified_code.replace("/submission.csv", f"/{submission_file_name}")

        if "to_csv('submission.csv" in modified_code:
            modified_code = modified_code.replace("to_csv('submission.csv", f"to_csv('submission/{submission_file_name}")
        if 'to_csv("submission.csv' in modified_code:
            modified_code = modified_code.replace('to_csv("submission.csv', f'to_csv("submission/{submission_file_name}')

        if '"submission.csv"' in modified_code:
            modified_code = modified_code.replace('"submission.csv"', f'"{submission_file_name}"')
        if "'submission.csv'" in modified_code:
            modified_code = modified_code.replace("'submission.csv'", f"'{submission_file_name}'")
        
        return modified_code
    
    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue, process_id: int
    ) -> None:
        
        self.child_proc_setup(result_outq)
        
        global_scope: dict = {}
        global_scope["__name__"] = "__main__"
        while True:
            
            code, id = code_inq.get()
            os.chdir(str(self.working_dir))
            cpu_number_per_session = int(self.cpu_number / self.max_parallel_run)
            start_cpu_id_session = self.start_cpu_id + process_id * cpu_number_per_session
            cpu_set = set()
            for i in range(start_cpu_id_session,start_cpu_id_session+cpu_number_per_session):
                cpu_set.add(i)
            logger.info(f"has set process_id:{process_id} to use cpu: {cpu_set}")
            pre_code = "import os\nos.sched_setaffinity(0, {cpu_set})\n".format(cpu_set=cpu_set)
            
            code  = self.replace_submission_name(code=code, _id=id)
            code = pre_code + code
            with open(self.agent_file_name[process_id], "w") as f:
                f.write(code)
            event_outq.put(("state:ready",))
            try:               
                exec(compile(code, self.agent_file_name[process_id], "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name[process_id],
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)
                
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # remove the file after execution (otherwise it might be included in the data preview)
            os.remove(self.agent_file_name[process_id])

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")

    def create_process(self,process_id) -> None:
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        # trunk-ignore(mypy/var-annotated)
        print(f"create process for {process_id}")
        self.code_inq[process_id], self.result_outq[process_id], self.event_outq[process_id] = Queue(), Queue(), Queue()
        self.process[process_id] = Process(
            target=self._run_session,
            args=(self.code_inq[process_id], self.result_outq[process_id], self.event_outq[process_id],process_id),
        )
        self.process[process_id].start()

    def cleanup_session(self,process_id):
        if process_id == -1 : #clean all process
            for pid in range(self.max_parallel_run):
                if self.process[pid] is None:
                    return
                # give the child process a chance to terminate gracefully
                self.process[pid].terminate()
                self.process[pid].join(timeout=2)
                # kill the child process if it's still alive
                if self.process[pid].exitcode is None:
                    logger.warning("Child process failed to terminate gracefully, killing it..")
                    self.process[pid].kill()
                    self.process[pid].join()
                # don't wait for gc, clean up immediately
                self.process[pid].close()
                self.process[pid] = None  # type: ignore
        else: #clean given process
            if self.process[process_id] is None:
                return
            # give the child process a chance to terminate gracefully
            self.process[process_id].terminate()
            self.process[process_id].join(timeout=2)
            # kill the child process if it's still alive
            if self.process[process_id].exitcode is None:
                logger.warning("Child process failed to terminate gracefully, killing it..")
                self.process[process_id].kill()
                self.process[process_id].join()
            # don't wait for gc, clean up immediately
            self.process[process_id].close()
            self.process[process_id] = None  # type: ignore

    def run(self, code: str, id, reset_session=True):
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.

        """
        logger.info(f"REPL is executing code (reset_session={reset_session})")
        logger.info(f"Current running process: {self.current_parallel_run}")
        process_id = None 
        # assign process_id
        with self.lock:
            self.current_parallel_run += 1
            for idx in range(self.max_parallel_run):
                if self.status_map[idx] == 0:
                    self.status_map[idx] = 1 # signals occupied
                    process_id = idx
                    logger.info(f"has assigned process_id，process_id is {process_id}")
                    break
                elif idx == self.max_parallel_run-1: # if not assigned
                    logger.info("reach max process parallel number")
                    raise ValueError("reach max process parallel number")
        
        if reset_session:
            if self.process[process_id] is not None:
                # terminate and clean up previous process
                try:
                    self.cleanup_session(process_id=process_id)
                except Exception as e:
                    logger.warning('reset_session cause a  bug in self.cleanup_session, the full traceback is:')
                    error_message = traceback.format_exc()
                    logger.warning(error_message) # Most likely, this process has already been killed, and killing the same process repeatedly has caused it to run normally

            self.create_process(process_id=process_id)
        else:
            # reset_session needs to be True on first exec
            assert self.process[process_id] is not None

        assert self.process[process_id].is_alive()

        self.code_inq[process_id].put((code, id))
        # wait for child to actually start execution (we don't want interrupt child setup)
        try:
            state = self.event_outq[process_id].get(timeout=10)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            logger.critical(msg)
            queue_dump = ""
            while not self.result_outq[process_id].empty():
                queue_dump = self.result_outq[process_id].get()
                logger.error(f"REPL output queue dump: {queue_dump[:1000]}")
            self.cleanup_session(process_id=process_id) 
            self.current_parallel_run -= 1 # Although it's a timeout, we still need to clean up the process table
            self.status_map[process_id] = 0
            return ExecutionResult(  # Considered as running timeout
                term_out=[msg, queue_dump],
                exec_time=0,
                exc_type="RuntimeError",
                exc_info={},
                exc_stack=[],
            )
        assert state[0] == "state:ready", state
        start_time = time.time()

        # this flag indicates that the child ahs exceeded the time limit and an interrupt was sent
        # if the child process dies without this flag being set, it's an unexpected termination
        child_in_overtime = False
        while True:
            try:
                # check if the child is done
                state = self.event_outq[process_id].get(timeout=1)  # wait for state:finished
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                # we haven't heard back from the child -> check if it's still alive (assuming overtime interrupt wasn't sent yet)
                if not child_in_overtime and not self.process[process_id].is_alive():
                    msg = "REPL child process died unexpectedly"
                    logger.critical(msg)
                    queue_dump = ""
                    while not self.result_outq[process_id].empty():
                        queue_dump = self.result_outq[process_id].get()
                        logger.error(f"REPL output queue dump: {queue_dump[:1000]}")
                    self.cleanup_session(process_id=process_id)
                    self.current_parallel_run -= 1 # Although it's a timeout, we still need to clean up the process table
                    self.status_map[process_id] = 0
                    return ExecutionResult(
                        term_out=[msg, queue_dump],
                        exec_time=0,
                        exc_type="RuntimeError",
                        exc_info={},
                        exc_stack=[],
                    )

                # child is alive and still executing -> check if we should sigint..
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:

                    # [TODO] handle this in a better way
                    assert reset_session, "Timeout ocurred in interactive session"

                    # send interrupt to child
                    os.kill(self.process[process_id].pid, signal.SIGINT)  # type: ignore
                    child_in_overtime = True
                    # terminate if we're overtime by more than a minute
                    if running_time > self.timeout + 60:
                        logger.warning("Child failed to terminate, killing it..")
                        self.cleanup_session(process_id=process_id)

                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        output: list[str] = []
        # read all stdout/stderr from child up to the EOF marker
        # waiting until the queue is empty is not enough since
        # the feeder thread in child might still be adding to the queue
        while not self.result_outq[process_id].empty() or not output or output[-1] != "<|EOF|>":
            res = self.result_outq[process_id].get()
            output.append(res)
            
        output.pop()  # remove the EOF marker

        e_cls_name, exc_info, exc_stack = state[1:]

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        else:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
            )
        print("execution done",ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack))
        self.current_parallel_run -= 1
        self.status_map[process_id] = 0
        self.cleanup_session(process_id=process_id) # Clean up the process after running it
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)





