import shutil
import logging
import random
import os
import time
from typing import cast
import math
import humanize
from backend import FunctionSpec, query, r1_query
from interpreter.interpreter_parallel import ExecutionResult
from search.mcts_node import MCTSNode
from utils.metric import MetricValue, WorstMetricValue
from utils.response import  wrap_code, extract_review
from utils.server_utils import call_validate


logger = logging.getLogger("ml-master")
review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "true if the code saves the predictions on the test data"
                " in a CSV file named either `submission.csv` or matching the pattern `submission_<hash>.csv` in the `./submission/` directory, otherwise false."
                " Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true."
                " Otherwise, it should be evaluated as false."
                " You can assume the ./submission/ directory exists and is writable.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (2-3 sentences) describing "
                " the empirical findings. Alternatively mention if there is a bug or"
                " the submission.csv was not properly produced."
                " DO NOT suggest fixes or improvements.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": [
            "is_bug",
            "has_csv_submission",
            "summary",
            "metric",
            "lower_is_better",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)

class FeedbackAgent:
    def parse_exec_result(self, node: MCTSNode, exec_result: ExecutionResult) -> MCTSNode:
        try:
            logger.info(f"Agent is parsing execution results for node {node.id}")

            node.absorb_exec_result(exec_result)

            introduction = (
                "You are a Kaggle grandmaster attending a competition. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            )
            if self.acfg.obfuscate:
                introduction = (
                    "You are an expert machine learning engineer attempting a task. "
                    "You have written code to solve this task and now need to evaluate the output of the code execution. "
                    "You should determine if there were any bugs as well as report the empirical findings."
                )
            prompt = {
                "Introduction": introduction,
                "Task description": self.task_desc,
                "Implementation": wrap_code(node.code),
                "Execution output": wrap_code(node.term_out, lang=""),
            }

            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=review_func_spec,
                    model=self.acfg.feedback.model,
                    temperature=self.acfg.feedback.temp,
                    convert_system_to_user=self.acfg.convert_system_to_user,
                    cfg=self.cfg
                ),
            )

            # if the metric isn't a float then fill the metric with the worst metric
            if not isinstance(response["metric"], float):
                response["metric"] = None

            # do an extra check, to catch cases where judge fails
            has_csv_submission = (
                self.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
            ).exists()

            node.analysis = response["summary"]
            if response["is_bug"] or node.exc_type is not None or response["metric"] is None or response["has_csv_submission"] == False or has_csv_submission == False:
                if response["is_bug"]:
                    logger.warning(f"Node {node.id} is marked as buggy because the response['is_bug'] is True.")
                elif node.exc_type is not None:
                    logger.warning(f"Node {node.id} is marked as buggy because the node.exc_type is not None.")
                elif response["metric"] is None:
                    logger.warning(f"Node {node.id} is marked as buggy because response['metric'] is None.")
                elif response["has_csv_submission"] == False:
                    logger.warning(f"Node {node.id} is marked as buggy because response['has_csv_submission'] is None.")
                else:
                    logger.warning(f"Node {node.id} is marked as buggy because has_csv_submission is False.")

            node.is_buggy = (
                response["is_bug"]
                or node.exc_type is not None
                or response["metric"] is None
                or has_csv_submission == False
            )
            if not node.is_buggy and self.acfg.check_format:
                exp_id = self.cfg.exp_name.split("_")[0]
                logger.info(f"Start checking the format of submission.csv of node {node.id}")
                status, res = call_validate(exp_id=exp_id, submission_path=self.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv")
                if status:
                    if not res['is_valid']:
                        logger.warning(f"Node {node.id} is marked as buggy because file: submission.csv is invalid.")
                        node.is_valid = False
                        node._term_out.append(f"\n{res['result']}")
                        node.analysis = "This previous solution runs without any bugs, but the format of the generated submission file is incorrect."
                    else:
                        node.is_valid = True
                        logger.info(f"Node {node.id} file: submission.csv is valid.")
                else:
                    logger.error(f"An unexpected error occurred: {res}, skip this stage.")
                    node.is_valid = True # set is_valid to True as default if using server is set but we can not connext to the server

            if node.is_buggy:
                logger.info(
                    f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
                )
                node.metric = WorstMetricValue()
            else:
                logger.info(f"Parsed results: Node {node.id} is not buggy")
                node.metric = MetricValue(
                    response["metric"], maximize=not response["lower_is_better"]
                )
            return node
        except Exception as e:
            logger.warning(f"parse result with tool error:{e}")
            logger.info("parse_exec_result_without_tool")
            return self.parse_exec_result_without_tool(node, exec_result)

    def parse_exec_result_without_tool(self, node: MCTSNode, exec_result: ExecutionResult) -> MCTSNode:
        logger.info(f"Agent is parsing execution results for node {node.id} without using tool.")
        node.absorb_exec_result(exec_result)
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings.\n\n"
            "You shoule evaluate the output of the code in Implementation. The review must be submitted in a specific JSON format with the following fields:\n\n"
            "- is_bug (boolean): This field is used to indicate whether any errors occurred during execution. If the output log shows that the execution failed or encountered a bug, set this value to true. Otherwise, set it to false.\n"
            "- has_csv_submission (boolean): This field indicates whether a submission CSV file was generated. If the code saves the predictions in a file named submission.csv in the ./submission/ directory, and it meets the required conditions, set this value to true. Otherwise, set it to false. Note that the file must be saved in the ./submission/ directory, and the filename may include a timestamp.\n"
            "- summary (string): In this field, provide a brief summary (2-3 sentences) describing the empirical findings. Alternatively, mention if there was a bug or if the submission.csv file was not properly produced. Do not suggest any fixes or improvements.\n"
            "- metric (number): If the code ran successfully, report the value of the validation metric here. If the code failed, this field should be set to null.\n"
            "- lower_is_better (boolean): This field indicates whether the metric should be minimized. If a lower value of the metric represents better performance (e.g., for Mean Squared Error), set this to true. If a higher value represents better performance (e.g., for accuracy), set this to false.\n\n"
            """The review must be submitted in the following JSON format in a single markdown code block (wrapped in ```):
```json
{
    "is_bug": true,  
    "has_csv_submission": false,  
    "summary": "The code encountered an error during execution. The CSV file was not generated.",
    "metric": null,  
    "lower_is_better": true  
}
```
"""
            ""
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            )
        prompt = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }
        try:
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
                cfg=self.cfg
            )
        except Exception as e:
            logger.info("parse without tool fail, try one more time.")
            completion_text = r1_query(
                prompt=prompt,
                temperature=self.acfg.code.temp,
                cfg=self.cfg
            )
        response = cast(
            dict,
            extract_review(completion_text)
        )
        if not isinstance(response["metric"], float):
            response["metric"] = None

        # do an extra check, to catch cases where judge fails
        has_csv_submission = (
            self.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
        ).exists()

        node.analysis = response["summary"]
        
        if response["is_bug"] or node.exc_type is not None or response["metric"] is None or response["has_csv_submission"] == False or has_csv_submission == False:
            if response["is_bug"]:
                logger.warning(f"Node {node.id} is marked as buggy because the response['is_bug'] is True.")
            elif node.exc_type is not None:
                logger.warning(f"Node {node.id} is marked as buggy because the node.exc_type is not None.")
            elif response["metric"] is None:
                logger.warning(f"Node {node.id} is marked as buggy because response['metric'] is None.")
            elif response["has_csv_submission"] == False:
                logger.warning(f"Node {node.id} is marked as buggy because response['has_csv_submission'] is None.")
            else:
                logger.warning(f"Node {node.id} is marked as buggy because has_csv_submission is False.")

        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
            or has_csv_submission == False
        )
        if not node.is_buggy and self.acfg.check_format:
            exp_id = self.cfg.exp_name.split("_")[0]
            logger.info(f"Start checking the format of submission.csv of node {node.id}")
            status, res = call_validate(exp_id=exp_id, submission_path=self.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv")
            if status:
                if not res['is_valid']:
                    logger.warning(f"Node {node.id} is marked as buggy because file: submission.csv is invalid.")
                    node.is_valid = False
                    node._term_out.append(f"\n{res['result']}")
                    node.analysis = "This previous solution runs without any bugs, but the format of the generated submission file is incorrect."
                else:
                    node.is_valid = True
                    logger.info(f"Node {node.id} file: submission.csv is valid.")
            else:
                logger.error(f"An unexpected error occurred: {res}, skip this stage.")

        if node.is_buggy:
            logger.info(
                f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
            )
            node.metric = WorstMetricValue()
        else:
            logger.info(f"Parsed results: Node {node.id} is not buggy")
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )
        return node    