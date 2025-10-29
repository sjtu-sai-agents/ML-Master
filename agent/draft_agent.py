import shutil
import logging
import random
import os
import time
from typing import Any, Callable, cast, Tuple, List, Literal
import math
import humanize
from backend import compile_prompt_to_md
from search.mcts_node import MCTSNode

logger = logging.getLogger("ml-master")
class DraftAgent:
    def _draft(self) -> MCTSNode:
        logger.info("Starting Drafting a new Node.")
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "In order to win this competition, you need to come up with an excellent and creative plan "
            "for a solution and then implement this solution in Python. We will now provide a description of the task."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "In order to complete this task, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.virtual_root.fetch_child_memory(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.\n",
                "- When proposing the design, take the Memory section into account.\n",
                "- In addition to incorporating the Memory module, it is **crucial** that your proposed solution **is distinctly different from** the existing designs in the Memory section.\n",
                "- Don't propose the same modelling solution but keep the evaluation the same.\n",
                "- The solution sketch should be 3-5 sentences.\n",
                "- Propose an evaluation metric that is reasonable for this task.\n",
                "- Don't suggest to do EDA.\n",
                "- The data is already prepared and available in the `./input` directory. There is no need to unzip any files.\n",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        instructions = f"\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)
        user_prompt = f"""
# Task description
{prompt['Task description']}

# Memory
The memory of previous solutions used to solve task is provided below:
{prompt['Memory']}

{instructions}
"""
        if "qwen3" in self.acfg.code.model and self.acfg.steerable_reasoning== True:
            prompt_complete = f"""<|im_start|>system
{introduction}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
<think>Okay! Now, I will focus my efforts on successfully completing this current task.
Before completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: 
{self.data_preview} 
"""        
        elif "deepseek" in self.acfg.code.model and self.acfg.steerable_reasoning== True:
            prompt_complete = f"""<｜begin▁of▁sentence｜>
{introduction}
<｜User｜>{user_prompt}<｜Assistant｜><think>
Okay! Now, I will focus my efforts on successfully completing this current task.
Before completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: 
{self.data_preview}
"""
        elif "gpt-5" in self.acfg.code.model or self.acfg.steerable_reasoning == False:
            user_prompt = f"""
# Task description
{prompt['Task description']}

# Memory
The memory of previous solutions used to solve task is provided below:
{prompt['Memory']}

{instructions}

# Data preview
{self.data_preview} 
"""
            prompt_complete = [
                    {"role": "system", "content": prompt['Introduction']},
                    {"role": "user", "content": user_prompt}
            ]
            
        self.virtual_root.add_expected_child_count()
        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(plan=plan, code=code, parent=self.virtual_root, stage="draft", local_best_node=self.virtual_root)
        logger.info(f"Drafted a new node {new_node.id} successfully!")
        return new_node