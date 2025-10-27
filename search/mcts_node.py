from search.node import Node
import math
from dataclasses import dataclass, field
from utils.config_mcts import SearchConfig
import uuid
import copy
import logging
from typing import Literal, Optional
import threading
import time
logger = logging.getLogger("ml-master")

@dataclass(eq=False)
class MCTSNode(Node):
    visits: int = field(default=0, kw_only=True)
    total_reward: float = field(default=0.0, kw_only=True)
    is_terminal: bool = field(default=False, kw_only=True)
    num_draft: int = field(default=5, kw_only=True) # TODO: hardcode
    num_nodes: int = field(default=3, kw_only=True) # TODO: hardcode
    _uct: float = field(default=0.0, kw_only=True)
    local_best_node: Optional["MCTSNode"] = field(default=None, kw_only=True)
    is_debug_success: bool = field(default=False, kw_only=True)
    continue_improve: bool = field(default=False, kw_only=True)
    stage: Literal["root", "improve", "debug", "draft"]
    improve_failure_depth: int = field(default=0, kw_only=True)
    lock: bool = field(default=False, kw_only=True)
    child_count_lock: bool = threading.Lock()
    expected_child_count: int = field(default=0, kw_only=True)
    finish_time: str = field(default=None, kw_only=True)


    def __post_init__(self):
        super().__post_init__()
        if self.stage not in ["root", "improve", "debug", "draft"]:
            raise ValueError(f"Invalid stage: {self.stage}")

    def uct_value(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate the UCT (Upper Confidence Bound for Trees) value of the current node.
        UCT = Q + c * sqrt(ln(N) / n), where:
        - Q = total_reward / visits (average reward)
        - c = exploration_constant (exploration constant, default is sqrt(2))
        - N = parent_visits (number of visits to the parent node)
        - n = visits (number of visits to the current node)
        """
        parent_visits: int= None
        if self.parent:
            parent_visits = self.parent.visits
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have the highest priority
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * (math.log(parent_visits) / self.visits) ** 0.5
        self._uct = exploitation + exploration
        # logger.info(f"Node {self.id} uct = {self.total_reward}/{self.visits} + {exploration_constant}*(math.log({parent_visits}) / {self.visits}) ** 0.5")
        return self._uct

    def is_fully_expanded(self, scfg: SearchConfig) -> bool:
        '''Different expansion strategies can be designed for different types of nodes:
        Draft node: 5
        Bug node: Stop expanding after obtaining a non-buggy node, with a maximum of 3
        Others: 3
        '''
        if self.step == 0:
            return self.num_children >= scfg.num_drafts
        else:
            if self.is_buggy:
                if self.has_no_bug_child():
                    return True
                else:
                    return self.num_children >= scfg.num_bugs
            else:
                return self.num_children >= scfg.num_improves
            
    def is_fully_expanded_with_expected(self, scfg: SearchConfig) -> bool:
        with self.child_count_lock:
            if self.step == 0:
                return self.expected_child_count >= scfg.num_drafts
            else:
                if self.is_buggy:
                    if self.has_no_bug_child():
                        return True
                    else:
                        return self.expected_child_count >= scfg.num_bugs
                else:
                    return self.expected_child_count >= scfg.num_improves

    
    def get_children_size(self) -> int:
        return len(self.children)

    def update(self, result, add=True):
        if add:
            self.visits += 1
            self.total_reward += result
        
    def has_no_bug_child(self):
        for child in self.children:
            if not child.is_buggy:
                return True
        return False

    @property
    def num_children(self):
        return len(self.children)

    def fetch_child_memory(self, include_code=False):
        logger.info("fetch_child_memory")
        summary = []
        for n in self.children:
            if n.is_buggy is not None:
                summary_part = f"Design: {n.plan}\n"
                if include_code:
                    summary_part += f"Code: {n.code}\n"
                if n.is_buggy is True:
                    summary_part += f"Results: The implementation of this design has bugs.\n"
                    summary_part += f"Insight: Using a different approach may not result in the same bugs as the above approach.\n"
                else:
                    if n.analysis:
                        summary_part += f"Results: {n.analysis}\n"
                    if n.metric:
                        summary_part += f"Validation Metric: {n.metric.value}\n"
                summary.append(summary_part)
        if len(summary) == 0:
            summary.append("There is no previous memory")
        return "\n-------------------------------\n".join(summary)

    def fetch_parent_memory(self, include_code=False):
        logger.info("fetch_parent_memory")
        if self.parent is not None and self.parent.is_buggy is not None and self.parent.is_buggy is False:
            summary = []
            summary_part = f"Design: {self.parent.plan}\n"
            if include_code:
                summary_part += f"Code: {self.parent.code}\n"
            summary_part += f"Results: {self.parent.analysis}\n"
            summary_part += f"Validation Metric: {self.parent.metric.value}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)
    
    def add_expected_child_count(self):
        with self.child_count_lock:
            self.expected_child_count += 1
            logger.info(f"current {self.id} expected_child_count is {self.expected_child_count}.")
    def sub_expected_child_count(self):
        with self.child_count_lock:
            self.expected_child_count -= 1
            logger.info(f"current {self.id} expected_child_count is {self.expected_child_count}.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('child_count_lock', None) 
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.child_count_lock = threading.Lock()
    
if __name__ == "__main__":
    MCTSNode()