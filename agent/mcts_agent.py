import shutil
import logging
import random
import os
import time
from typing import Callable
import math
import humanize
from interpreter.interpreter_parallel import ExecutionResult
from search.journal import Journal
from search.mcts_node import MCTSNode
from utils.config_mcts import Config
from utils.metric import  WorstMetricValue
from utils.mcts import linear_decay, exponential_decay, piecewise_decay, dynamic_piecewise_decay
import threading
from agent.agent_utils import AgentUtils
from agent.debug_agent import DebugAgent
from agent.improve_agent import ImproveAgent
from agent.feedback_agent import FeedbackAgent
from agent.draft_agent import DraftAgent

logger = logging.getLogger("ml-master")
ExecCallbackType = Callable[[str, bool], ExecutionResult]

class MCTSAgent(AgentUtils,DebugAgent,ImproveAgent,FeedbackAgent,DraftAgent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.scfg = cfg.agent.search
        self.journal = journal
        self.data_preview: str | None = None
        self.current_step = 0
        self.current_node: MCTSNode | None = None
        self.all_root = True
        self.virtual_root = MCTSNode(parent=None, plan="virtual plan", code="# virtual code", metric=WorstMetricValue(), stage="root")
        self.current_node_list = []
        self.journal.append(self.virtual_root)
        self.best_metric: float = None
        self.best_node: MCTSNode = None
        self.search_start_time = None
        self.journal_lock = threading.Lock()
        self.save_node_lock = threading.Lock()
        self.start_time = time.time()

    def backpropagate(self, node: MCTSNode, value: float, add_to_tree=True):
        logger.info(f"node {node.id} start backpropagating with reward {value}.")
        while node != None:
            if node.is_buggy is False and node.parent.is_buggy is True:
                node.parent.is_debug_success = True
            elif node.is_buggy is True and node.is_debug_success is True and node.parent.is_buggy is True:
                node.parent.is_debug_success = True
            if node.parent and node.parent.stage != "root":
                node.parent.continue_improve = node.continue_improve
            if node.stage == "draft" and node.lock:
                node.lock = False
                logger.info(f"Draft node {node.id} is unlocked.")
            if node.improve_failure_depth>0:
                node.improve_failure_depth = 0
            node.update(value, add_to_tree)
            node = node.parent

    def select(self, node: MCTSNode):
        logger.info(f"[select] Processing node: {node.id}")
        while node and not node.is_terminal:
            if not node.is_fully_expanded_with_expected(scfg=self.scfg):
                if node.is_buggy and node.is_debug_success is True:
                    node = self.uct_select(node)
                elif node.continue_improve and len(node.children)>0:
                    node = self.uct_select(node)
                else:
                    logger.info(f"Node {node.id} is not fully expanded, expanding")
                    return node
            else:
                node = self.uct_select(node)
        logger.info(f"[select]choose a node for expanding: {node.id}")
        return node

    def get_C(self):
        dcfg =  self.cfg.agent.decay
        if dcfg.decay_type == "linear":
            linear_cfg = dcfg.linear_decay
            return linear_decay(
                t=self.current_step, 
                initial_C=dcfg.exploration_constant,
                lower_bound=dcfg.lower_bound,
                alpha=linear_cfg.alpha
            )
        
        elif dcfg.decay_type == "exponential":
            exponential_cfg = dcfg.exponential_decay
            return exponential_decay(
                t=self.current_step,
                initial_C=self.scfg.exploration_constant,
                lower_bound=dcfg.lower_bound,
                gamma=exponential_cfg.gamma,
            )
        
        elif dcfg.decay_type == "piecewise":
            piecewise_cfg = dcfg.piecewise_decay
            n1 = self.scfg.num_drafts*(self.scfg.num_improves ** 2)
            n2 = round(self.acfg.steps*piecewise_cfg.phase_ratios[0])
            t1 = min(n1,n2)
            t2 = round(self.acfg.steps*piecewise_cfg.phase_ratios[1])
            return piecewise_decay(
                t=self.current_step, 
                initial_C=dcfg.exploration_constant,
                T1=t1,
                T2=t2,
                lower_bound=dcfg.lower_bound
            )
        
        elif dcfg.decay_type == "dynamic_piecewise":
            dynamic_piecewise_cfg = dcfg.dynamic_piecewise_decay
            logger.info(f"dynamic_piecewise_cfg.phase_ratios = {dynamic_piecewise_cfg.phase_ratios}")
            return dynamic_piecewise_decay(
                steps_limit=self.acfg.steps,
                n_nodes=self.current_step,
                initial_C=dcfg.exploration_constant,
                start_time=self.search_start_time,
                time_limit=self.acfg.time_limit,
                alpha=dynamic_piecewise_cfg.alpha,
                lower_bound=dcfg.lower_bound,
                phase_ratios=dynamic_piecewise_cfg.phase_ratios
            )
        else:
            return dcfg.exploration_constant

    def uct_select(self, node: MCTSNode):
        if self.is_root(node):
            filtered_children = [child for child in node.children if not child.lock]
            logger.info(f"For node {node.id}, there are {len(node.children) - len(filtered_children)}/{len(node.children)} is locked.")
            selected_node = node
            if len(filtered_children) > 0:
                selected_node = max(filtered_children, key=lambda child: child.uct_value(exploration_constant = self.get_C()))
                
            if selected_node.stage == "draft":
                selected_node.lock = True
                logger.info(f"Draft node {selected_node.id} is locked.")
            return selected_node
        else:
            return max(node.children, key=lambda child: child.uct_value(exploration_constant = self.get_C()))

    
    def check_improvement(self, cur_node: MCTSNode, parent_node: MCTSNode):
        improvement = 0
        should_backpropagate = False
        local_best_node = cur_node.local_best_node
        local_best_metric = local_best_node.metric.value
        if cur_node.is_buggy is False:
            new_metric = cur_node.metric.value  
            if parent_node.is_buggy:
                logger.info(f"Successfully Debug the error in node {parent_node.id}.")
                should_backpropagate = True
            if new_metric and local_best_metric:
                improvement = new_metric - local_best_metric if cur_node.metric.maximize else local_best_metric - new_metric
                if improvement < self.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth < self.scfg.max_improve_failure:
                    local_best_node.improve_failure_depth += 1
                    logger.warning(f"Compared to Node {local_best_node.id}, Node {cur_node.id} metric improvement ({improvement}) below threshold ({self.scfg.metric_improvement_threshold}), try one more time({local_best_node.improve_failure_depth}/{self.scfg.max_improve_failure})")
                    cur_node.continue_improve = True
                elif improvement < self.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth >= self.scfg.max_improve_failure:
                    logging.warning(f"The number of improvement attempts for the local best node has reached its maximum limit {self.scfg.max_improve_failure}.")
                    cur_node.continue_improve = False
                    should_backpropagate = True
                    cur_node.is_terminal = True
                else:
                    logger.info(f"Compared to Node {local_best_node.id}, Node {cur_node.id} metric improvement ({improvement}) above threshold ({self.scfg.metric_improvement_threshold}), continue improving.")
                    cur_node.local_best_node = cur_node
                    cur_node.continue_improve = True
            elif new_metric:
                logger.info(f"No local best node was found among the previous nodes; the current node {cur_node.id} is assigned as the local best")
                cur_node.local_best_node = cur_node
                cur_node.continue_improve = True
            else:
                logger.warning(f"No local best node was found among the previous nodes; The current node {cur_node.id} has no errors, but contains an empty metric value.")
                should_backpropagate = True
        elif cur_node.is_buggy is None:
            logger.warning(f"Node {cur_node.id} is_buggy is None!")
            should_backpropagate = True
        else:
            if cur_node.debug_depth >= self.scfg.back_debug_depth:
                should_backpropagate = True
                if cur_node.debug_depth >= self.scfg.max_debug_depth:
                    cur_node.is_terminal = True

        if should_backpropagate:
            reward = self.get_node_reward(cur_node)
            self.backpropagate(cur_node, reward)
        else:
            self.current_node_list.append(cur_node)
        return should_backpropagate
    
    def get_node_reward(self, node: MCTSNode):
        reward = 0
        if node.is_buggy is True or node.is_buggy is None:
            reward = -1
        elif node.is_buggy is False and node.metric.value is None:
            reward = -1
        else:
            if node.metric.value and self.best_metric:
                improvement = node.metric.value - self.best_metric if node.metric.maximize else self.best_metric - node.metric.value
                if improvement > 0:
                    logger.info(f"Node {node.id} is better than the best node {self.best_node.id} now!")
                    reward += 1
            if node.parent.is_buggy is True:
                reward += 1
            else:
                reward += 1
        return reward
            
    def is_root(self, node: MCTSNode):
        return node.id is self.virtual_root.id
    
    def check_metric_valid(self, node: MCTSNode, upper_bound=50):
        '''If the metric values between nodes differ by an upper bound multiple, it is highly likely that there is an invalid metric'''
        upper_bound = self.acfg.search.invalid_metric_upper_bound if self.acfg.search.invalid_metric_upper_bound else upper_bound
        v1 = self.best_metric
        v2 = node.metric.value
        if v1 is None or v2 is None:
            return True
        elif v1 == 0 or v2 == 0:
            return abs(v1 - v2) <= upper_bound
        else:
            ratio = max(abs(v1), abs(v2)) / min(abs(v1), abs(v2))
            return ratio <= upper_bound

    def _step_search(self, parent_node: MCTSNode, exec_callback: ExecCallbackType):
        logger.info(f"[_step_search] Processing node: {parent_node.id}")
        logger.info(f"Agent is generating code, parent node type: {type(parent_node)}")
        result_node = None
        _root = False
    
        if not parent_node.is_terminal:
            try:
                if self.is_root(parent_node):
                    result_node = self._draft()
                    result_node.lock = True
                    logger.info(f"[_step_search]Draft node {result_node.id} is locked.")
                elif parent_node.is_buggy or parent_node.is_valid is False:
                    result_node = self._debug(parent_node)
                elif parent_node.is_buggy is False:
                    result_node = self._improve(parent_node)
                else:
                    logger.warning(f"[_step_search] node {parent_node.id} is_buggy is None.")
                
                if result_node:
                
                    exe_res = exec_callback(result_node.code, result_node.id, True)
                
                    result_node = self.parse_exec_result(
                        node=result_node,
                        exec_result=exe_res
                    )
                    if not result_node.is_buggy:
                        if not (self.cfg.workspace_dir / "submission" / f"submission_{result_node.id}.csv").exists():
                            result_node.is_buggy = True
                            result_node.metric = WorstMetricValue()
                            logger.info(f"Actually, node {result_node.id} did not produce a submission.csv")
                    logger.info(f"The metric value of node {result_node.id} is {result_node.metric.value}.")
                    if not self.check_metric_valid(node=result_node):
                        result_node.metric = WorstMetricValue()
                        logger.info(f"node {result_node.id} generate invalid metric.")
                    result_node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")
                    if parent_node.is_buggy and result_node.is_buggy is False:
                        parent_node.is_debug_success = True
                    
                    _root = self.check_improvement(result_node, parent_node)
                    with self.journal_lock:
                        if self.best_node and result_node.metric.maximize and self.best_node.metric.maximize != result_node.metric.maximize:
                            logger.warning("New node's metric is inconsistent with metrics in the journal.Returning to the parent node to regenerate.")
                            raise ValueError("New node's metric is inconsistent with metrics in the journal.Returning to the parent node to regenerate.")
                        else:
                            self.journal.append(result_node)
                            

            except Exception as e:
                logger.warning("Current node generation failed, rolling back to unlock the draft node.")
                self.backpropagate(node=parent_node, value=0, add_to_tree=False)
                parent_node.sub_expected_child_count()
                raise e

        else:
            logger.info(f"current node is terminal, backpropagating!!")
            self.backpropagate(node=parent_node, value=0)
            _root = True
        return _root, result_node
    
    def get_best_node(self, node_list):
        good_node = [n for n in node_list if not n.is_buggy and n.metric]
        if not good_node:
            return None
        return max(good_node, key=lambda n: n.metric)

    def step(self, node: MCTSNode, exec_callback: ExecCallbackType) -> bool:   
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
            self.search_start_time = time.time()

        if not node or node.stage == "root":
            node = self.select(self.virtual_root)

        _root, result_node = self._step_search(node, exec_callback=exec_callback)
        if result_node:
            submission_file_path = self.cfg.workspace_dir / "submission" / f"submission_{result_node.id}.csv"
            logger.info(f"In the search step from node {node.id}, the generated node is {result_node.id}, the metric is {result_node.metric.value}")
        if result_node and result_node.metric.value is not None:
            if self.best_node is None or self.best_node.metric < result_node.metric:
                logger.info(f"Node {result_node.id} is the best node so far")
                if self.best_node is None or result_node.is_valid is True:
                    self.best_node = result_node
                    best_solution_dir = self.cfg.workspace_dir / "best_solution"
                    best_submission_dir = self.cfg.workspace_dir / "best_submission"
                    with self.save_node_lock:
                        best_solution_dir.mkdir(exist_ok=True, parents=True)
                        best_submission_dir.mkdir(exist_ok=True, parents=True)
                        shutil.copy(
                            submission_file_path,
                            best_submission_dir / "submission.csv",
                        )
                        with open(best_solution_dir / "solution.py", "w") as f:
                            f.write(result_node.code)
                        with open(best_solution_dir / "node_id.txt", "w") as f:
                            f.write(str(result_node.id))
                else:
                    logger.info(f"Node {result_node.id} is a invalid node")
                    logger.info(f"Node {self.best_node.id} is still the best node")
            else:
                if self.best_node.is_valid is False:
                    logger.info(f"Node {self.best_node.id} is invalid, {result_node.id} is the best node so far")
                    self.best_node = result_node
                    best_solution_dir = self.cfg.workspace_dir / "best_solution"
                    best_submission_dir = self.cfg.workspace_dir / "best_submission"
                    with self.save_node_lock:
                        best_solution_dir.mkdir(exist_ok=True, parents=True)
                        best_submission_dir.mkdir(exist_ok=True, parents=True)
                        shutil.copy(
                            submission_file_path,
                            best_submission_dir / "submission.csv",
                        )
                        with open(best_solution_dir / "solution.py", "w") as f:
                            f.write(result_node.code)
                        with open(best_solution_dir / "node_id.txt", "w") as f:
                            f.write(str(result_node.id))

                else:
                    logger.info(f"Node {result_node.id} is not the best node")
                    logger.info(f"Node {self.best_node.id} is still the best node")
        elif not result_node:
            logger.info(f"Result node is None.")
        else:
            logger.info(f"result node has bug.")
        if self.best_node:
            logger.info(f"Best metric value is {self.best_node.metric.value}.")

        if not self.acfg.save_all_submission and result_node and os.path.exists(submission_file_path):
            os.remove(submission_file_path)
        self.current_step = len(self.journal)
        if _root or result_node is None:
            logger.info(f"agent return root to main")
            return self.virtual_root
        else:
            logger.info(f"agent return {result_node.id} to main")
            return result_node