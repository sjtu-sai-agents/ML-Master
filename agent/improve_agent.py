import shutil
import logging
import random
import os
import time
from typing import Any
import math
import humanize
from backend import compile_prompt_to_md
from search.mcts_node import MCTSNode

from utils.response import  wrap_code

logger = logging.getLogger("ml-master")

class ImproveAgent:
    def _improve(self, parent_node: MCTSNode) -> MCTSNode:
        logger.info(f"Starting Improving Node {parent_node.id}.")
        introduction = (
            "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
            "solution below and should improve it in order to further increase the (test time) performance. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. "
        )

        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": parent_node.fetch_child_memory(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "- The solution sketch should be a brief natural language description of how the previous solution can be improved.\n",
                "- You should be very specific and should only propose a single actionable improvement.\n",
                "- This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.\n",
                "- When proposing the design, take the Memory section into account.\n",
                "- In addition to incorporating the Memory module, it is **crucial** that your proposed solution **is distinctly different from** the existing designs in the Memory section.\n",
                "- The solution sketch should be 3-5 sentences.\n",
                "- Don't suggest to do EDA.\n",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        output = wrap_code(parent_node.term_out, lang="")
        
        instructions = "\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)
        
        user_prompt = f"""
# Task description
{prompt['Task description']}
# Memory
The memory of previous solutions used to improve performance is provided below: 
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
Regarding this task, I previously made attempts with the following code:
{prompt['Previous solution']['Code']}
The execution of this code yielded the following results:
{output}
I believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance.
"""
        elif "deepseek" in self.acfg.code.model and self.acfg.steerable_reasoning== True:
            prompt_complete = f"""<｜begin▁of▁sentence｜>{introduction}
<｜User｜>{user_prompt}<｜Assistant｜><think>
Okay! Now, I will focus my efforts on successfully completing this current task.
Before completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: 
{self.data_preview}
Regarding this task, I previously made attempts with the following code:
{prompt['Previous solution']['Code']}
The execution of this code yielded the following results:
{output}
I believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance. 
"""
        elif "gpt-5" in self.acfg.code.model or self.acfg.steerable_reasoning == False:
            user_prompt = f"""
# Task description
{prompt['Task description']}
# Memory
The memory of previous solutions used to improve performance is provided below: 
{prompt['Memory']}

{instructions}

# Data preview
{self.data_preview}

# Previous solution
{prompt['Previous solution']['Code']}

# Execution output
{output}
"""
            prompt_complete = [
                    {"role": "system", "content": prompt['Introduction']},
                    {"role": "user", "content": user_prompt}
            ]
        parent_node.add_expected_child_count()

        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(plan=plan, code=code, parent=parent_node, stage="improve", local_best_node=parent_node.local_best_node)
        logger.info(f"Improving node {parent_node.id} to create new node {new_node.id}")
        return new_node