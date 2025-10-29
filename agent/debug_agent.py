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
class DebugAgent:
    def _debug(self, parent_node: MCTSNode) -> MCTSNode:
        logger.info(f"Starting Debugging Node {parent_node.id}.")
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug and/or did not produce a submission.csv, "
            "so based on the information below, you should revise it in order to fix this. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        )

        if self.acfg.check_format:
            introduction = (
                "You are a Kaggle grandmaster attending a competition. "
                "Your previous solution had a bug and/or did not produce a submission.csv, or the generated submission.csv was in an incorrect format,"
                "so based on the information below, you should revise it in order to fix this. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            )

        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "- You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.\n",
                "- Don't suggest to do EDA.\n",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        instructions = "\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)
        user_prompt = f"""
# Task description
{prompt['Task description']}

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
Regarding this task, I previously made an attempt with the following code:
{prompt['Previous (buggy) implementation']}
However, there are the following issues with this code:
{prompt['Execution output']}
I hold the view that the underlying reasons giving rise to the emergence of this issue are:
{parent_node.analysis}
The previous solution had a bug and/or did not produce a submission.csv. I will try to fix the bug.
"""
        elif "deepseek" in self.acfg.code.model and self.acfg.steerable_reasoning== True:
            prompt_complete = f"""<｜begin▁of▁sentence｜>
{prompt['Introduction']}
<｜User｜>{user_prompt}<｜Assistant｜><think>
Okay! Now, I will focus my efforts on successfully completing this current task.
Before completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: 
{self.data_preview}
Regarding this task, I previously made an attempt with the following code:
{prompt['Previous (buggy) implementation']}\nHowever, there are the following issues with this code:
{prompt['Execution output']}
I hold the view that the underlying reasons giving rise to the emergence of this issue are:
{parent_node.analysis}
The previous solution had a bug and/or did not produce a submission.csv, or the generated submission.csv was in an incorrect format.I will try to fix the bug.
"""
        elif "gpt-5" in self.acfg.code.model or self.acfg.steerable_reasoning == False:
            user_prompt = f"""
# Task description
{prompt['Task description']}

{instructions}

# Data preview
{self.data_preview}

# Previous (buggy) implementation
{prompt['Previous (buggy) implementation']}

# Execution output
{prompt['Execution output']}
"""
            prompt_complete = [
                    {"role": "system", "content": prompt['Introduction']},
                    {"role": "user", "content": user_prompt}
            ]        

        parent_node.add_expected_child_count()
        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(plan=plan, code=code, parent=parent_node, stage="debug", local_best_node=parent_node.local_best_node)
        logger.info(f"Debugging node {parent_node.id} to create new node {new_node.id}")
        return new_node