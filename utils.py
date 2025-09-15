import re
import ast
import json
import jedi
import autopep8
import logging
import numpy as np
from openai import OpenAI
from structured_logprobs.main import add_logprobs, add_logprobs_inline
from typing import List, Optional, Dict, Union, Tuple, Any
from pydantic import BaseModel, RootModel, Field
import copy
import itertools
from math import exp, log, inf, sqrt
from multi_process import monitor_process, make_serializable, is_serializable
# MCTS 超参数
MAX_EXECUTION_TIME = 10  # 秒
MAX_MEMORY_USAGE = 512  # MB

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key='sk-or-v1-5xxxxxxxxxxxxxxxxxxxxxxxxxxx')

# --- Pydantic Models ---
# Models are now simpler as they don't need to hold logprobs.
# The logprobs will be returned in a separate dictionary by `add_logprobs`.

class Task(BaseModel):
    """A model containing only a task description."""
    task: str = Field(..., description="A specific description of the subtask.")


class TaskDict(RootModel[Dict[str, Task]]):
    """A root model for a dictionary of tasks, used for initial decomposition."""
    root: Dict[str, Task]


class TaskCode(BaseModel):
    """A model for a task and a single generated piece of code."""
    task: str = Field(..., description="The description of the current task.")
    code: str = Field(..., description="A candidate Python code for the current task.")


class SubTask(BaseModel):
    """Defines a single subtask with a description and code."""
    task: str = Field(..., description="A specific description of the subtask.")
    code: str = Field("", description="The Python code to execute the subtask.")

class SubLogpro(BaseModel):
    task: float = Field(..., description="A logpro of the subtask.")

class TaskOutput(BaseModel):
    """The standard output structure for functions returning a set of tasks."""
    tasks: Dict[str, SubTask] = Field(..., description="A dictionary of subtasks, keyed by task ID.")
    transition_scores: Dict[str, SubLogpro] = Field(..., description="Confidence scores for each subtask.")


class CandidateCode(BaseModel):
    """Represents a single code candidate with its associated score."""
    code: str
    score: float

class TaskCodeWithCandidates(BaseModel):
    """The output structure for code generation with multiple candidates."""
    task: str
    candidates: List[CandidateCode]



# --- Helper Functions ---
# The manual score calculation is no longer needed.

def _get_average_logprob(logprobs_dict: Dict) -> float:
    """Calculates the average probability from a dictionary of logprobs."""
    if not logprobs_dict:
        return 0.0
    # Use np.exp to convert logprobs back to probabilities before averaging
    probs = [np.exp(lp) for lp in logprobs_dict.values() if isinstance(lp, (int, float))]
    return np.mean(probs) if probs else 0.0


# --- Core Functions Refactored with structured_logprobs ---
def decompose_task(user_task: str, pre_task: str = '') -> TaskOutput:
    """Decomposes a user task into subtasks using structured_logprobs."""
    system_prompt = "You are an assistant that decomposes a task into step-by-step executable subtasks."
    user_prompt = f""" Task: {user_task} Please break it down into executable steps and return as a numbered task list. 
    Return ONLY valid JSON with the following structure: 
    {{ 
    "0": {{"task": "the description of the first task"}}, 
    "1": {{"task": "the description of the second task"}},
    ... 
    }} 
    Do NOT add explanations or extra text. """
    if pre_task:
        user_prompt += f"\nThe task1 must be: {pre_task}"
    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3-8b:free",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            logprobs=True,
            temperature=0.7,
            #max_tokens=1500,  # 限制生成的 token 数量
            #stop = ["\n\n", "}"],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "task_output",
                    "schema": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"}
                            },
                            "required": ["task"]
                        }
                    }
                }
            }
        )
        chat_completion = add_logprobs(completion)
        #print(chat_completion)
        #print(completion.choices)
        raw_output = chat_completion.value.choices[0].message.content #finish_reason='stop'
        logprobs = chat_completion.log_probs[0]
        #print(raw_output)
        #print(logprobs)
        # 解析任务 JSON
        if isinstance(raw_output, str):
            try:
                task_dict = json.loads(raw_output)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON: {e}")
                return TaskOutput(tasks={}, transition_scores={})
        elif isinstance(raw_output, dict):
            task_dict = raw_output
        else:
            logging.error(f"Unexpected type for raw_output: {type(raw_output)}")
            return TaskOutput(tasks={}, transition_scores={})

        # 解析任务 JSON 并转换成 SubTask
        tasks_output = {}
        for k, v in task_dict.items():
            if isinstance(v, dict) and "task" in v:
                tasks_output[k] = SubTask(task=v["task"], code=v.get("code", ""))
            else:
                tasks_output[k] = SubTask(task=str(v))

        # 解析 logprobs 并转换成 SubLogpro
        transition_scores_output = {}
        for k, v in logprobs.items():
            if isinstance(v, dict) and "task" in v:
                transition_scores_output[k] = SubLogpro(task=v["task"])
            else:
                # 如果结构不同，可以设置默认值或处理异常
                transition_scores_output[k] = SubLogpro(task=float(v))

        return TaskOutput(tasks=tasks_output, transition_scores=transition_scores_output)
    except Exception as e:
        logging.error(f"An error occurred in decompose_task: {e}")
        return TaskOutput(tasks={}, transition_scores={})


def re_decompose_task(user_task: str, curr_task:str = '', pre_task: str = '') -> TaskOutput:
    """Decomposes a user task into subtasks using structured_logprobs."""
    system_prompt = "You are an assistant that resolve the difficult steps of the Task and decomposes it into easier step-by-step executable subtasks."
    user_prompt = f""" Task: {user_task} \n
    the current difficult task is: {curr_task}, which can not be solved after several trials \n
    Please skip this difficult step and break the rest Task down into EASIER executable steps and return as a numbered task list. 
    Return ONLY valid JSON with the following structure: 
    {{ 
    "0": {{"task": "the description of the first task"}}, 
    "1": {{"task": "the description of the second task"}},
    ... 
    }} 
    Do NOT add explanations or extra text. """
    if pre_task:
        user_prompt += f"\nThe task1 must be: {pre_task}"
    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3-8b:free",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            logprobs=True,
            temperature=0.7,
            #max_tokens=1500,  # 限制生成的 token 数量
            #stop = ["\n\n", "}"],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "task_output",
                    "schema": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"}
                            },
                            "required": ["task"]
                        }
                    }
                }
            }
        )
        chat_completion = add_logprobs(completion)
        #print(chat_completion)
        #print(completion.choices)
        raw_output = chat_completion.value.choices[0].message.content #finish_reason='stop'
        logprobs = chat_completion.log_probs[0]
        #print(raw_output)
        #print(logprobs)
        # 解析任务 JSON
        if isinstance(raw_output, str):
            try:
                task_dict = json.loads(raw_output)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON: {e}")
                return TaskOutput(tasks={}, transition_scores={})
        elif isinstance(raw_output, dict):
            task_dict = raw_output
        else:
            logging.error(f"Unexpected type for raw_output: {type(raw_output)}")
            return TaskOutput(tasks={}, transition_scores={})

        # 解析任务 JSON 并转换成 SubTask
        tasks_output = {}
        for k, v in task_dict.items():
            if isinstance(v, dict) and "task" in v:
                tasks_output[k] = SubTask(task=v["task"], code=v.get("code", ""))
            else:
                tasks_output[k] = SubTask(task=str(v))

        # 解析 logprobs 并转换成 SubLogpro
        transition_scores_output = {}
        for k, v in logprobs.items():
            if isinstance(v, dict) and "task" in v:
                transition_scores_output[k] = SubLogpro(task=v["task"])
            else:
                # 如果结构不同，可以设置默认值或处理异常
                transition_scores_output[k] = SubLogpro(task=float(v))

        return TaskOutput(tasks=tasks_output, transition_scores=transition_scores_output)
    except Exception as e:
        logging.error(f"An error occurred in decompose_task: {e}")
        return TaskOutput(tasks={}, transition_scores={})

def generate_task_codes(pre_tasks: Dict, curr_task: str, k: int = 3,
                        base_temperature: float = 0.7) -> TaskCodeWithCandidates:
    """Generates k code candidates for a task by calling the model k times with slight temperature perturbation."""
    import random
    pre_tasks = {k: {kk: vv for kk, vv in v.items() if kk != 'score'} for k, v in pre_tasks.items()}
    pre_tasks_str = json.dumps(pre_tasks, indent=2)
    system_prompt = """You are an expert Python programming assistant.
    Your sole purpose is to generate a Python code solution for a given task.
    You MUST strictly follow these output rules:
    1.  Your entire response must be ONLY a single, valid JSON object.
    2.  The Python code string must be properly escaped to be a valid JSON string value. This means newlines must be `\\n`, double quotes must be `\\"`, etc.
    """
    user_prompt = f"""Previous tasks context:\n{pre_tasks_str}\n\n
    Current task to implement: {curr_task}\n\n
    Generate one Python code solution for the current task.
    Return ONLY valid JSON with the following structure: 
    {{"code": "the Python code solution"}}
    Do NOT explain, do NOT add reasoning, do NOT add extra text. Only return the requested output.
    """

    candidates = []

    for attempt in range(k):
        # 随机扰动 temperature
        temperature = base_temperature + random.uniform(-0.1, 0.1)
        try:
            completion = client.chat.completions.create(
                model="qwen/qwen3-8b:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                logprobs=True,
                temperature=temperature,
                #max_tokens=512,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "task_code",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"}  # Python 代码
                            },
                            "required": ["code"],
                            "additionalProperties": False
                        }
                    }
                }
            )

            chat_completion = add_logprobs(completion)
            raw_output = chat_completion.value.choices[0].message.content
            logprobs = chat_completion.log_probs[0]
            #print(raw_output)
            #print(logprobs)
            # 尝试解析 JSON
            try:
                parsed_data = json.loads(raw_output)
            except json.JSONDecodeError:
                logging.warning(f"Attempt {attempt+1}: JSON parse failed, skipping candidate.")
                continue
            if 'code' in parsed_data:
                # 取 logprob，简单示例：用 'code' key 对应的 logprob 或默认 -inf
                score = float(logprobs.get('code', -np.inf))
                candidates.append(CandidateCode(code=parsed_data['code'], score=score))
                #print(candidates)
        except Exception as e:
            logging.error(f"Attempt {attempt+1}: error generating code: {e}")
            continue

    return TaskCodeWithCandidates(task=curr_task, candidates=candidates)


def llm_task_update(current_tasks: Dict, current_errors: str, temperature: float = 0.7) -> TaskOutput:
    """Refines a task list and generates code for each task using structured_logprobs."""

    # 删除 score，保持任务干净
    current_tasks_clean = {
        k: {kk: vv for kk, vv in v.items() if kk != 'score'}
        for k, v in current_tasks.items()
    }

    system_prompt = """You are an expert assistant specializing in Python programming task planning.
    Your core mission is to refine, split, or fix a list of tasks based on a plan and execution errors.
    You MUST strictly adhere to the following output rules:
    1.  Your entire response must be ONLY a single, valid JSON object. Do not add any reasoning, markdown, or text outside the JSON structure.
    2.  The Python code inside the "code" field must be a valid JSON string. Ensure all special characters are escaped (e.g., newlines as \\n, double quotes as \\", backslashes as \\\\).
    """
    user_prompt = f"""Here is the current task plan: {json.dumps(current_tasks_clean, indent=2)}\n
    Here are the errors encountered during execution: {current_errors}\n
    Based on the above, refine the task list by fixing or splitting into subtasks as needed.
    Return ONLY valid JSON with the following structure: 
    {{ 
    "0": {{"task": "the description of this or sub task", "code": "the python code for the task or subtask"}}, 
    "1": {{"task": "the description of this or sub task", "code": "the python code for the task or subtask"}},
    ...
    }} 
    """

    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3-8b:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            logprobs=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "updated_tasks",
                    "schema": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},  # 子任务描述
                                "code": {"type": "string"}  # Python 代码
                            },
                            "required": ["task", "code"]
                        }
                    }
                }
            }
        )

        chat_completion = add_logprobs(completion)
        raw_output = chat_completion.value.choices[0].message.content
        logprobs = chat_completion.log_probs[0]

        # 解析 JSON
        try:
            tasks_dict = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}")
            return TaskOutput(tasks={}, transition_scores={})

        # 构建 TaskOutput
        final_tasks = {}
        transition_scores = {}

        for k, v in tasks_dict.items():
            if isinstance(v, dict):
                final_tasks[k] = SubTask(
                    task=v.get("task", ""),
                    code=v.get("code", "")
                )
            else:
                final_tasks[k] = SubTask(task=str(v), code="")

            # logprobs 转 score
            if k in logprobs and isinstance(logprobs[k], dict) and "task" in logprobs[k]:
                transition_scores[k] = SubLogpro(task=float(logprobs[k]["task"]))
            else:
                transition_scores[k] = SubLogpro(task=0.0)

        return TaskOutput(tasks=final_tasks, transition_scores=transition_scores)

    except Exception as e:
        logging.error(f"An error occurred in llm_task_update: {e}")
        return TaskOutput(tasks={}, transition_scores={})


def generate_full_task_code(pre_tasks: Dict, curr_tasks: Dict, temperature: float = 0.7) -> TaskOutput:
    """Generates full code for current tasks given previous tasks context, using structured_logprobs."""
    pre_tasks = {k: {kk: vv for kk, vv in v.items() if kk != 'score'} for k, v in pre_tasks.items()}
    curr_tasks = {k: {kk: vv for kk, vv in v.items() if kk != 'score'} for k, v in curr_tasks.items()}
    pre_tasks_str = json.dumps(pre_tasks, indent=2)
    system_prompt = """You are an expert Python programmer.
    Your mission is to complete the 'code' for each task provided by the user.
    You MUST adhere to the following rules for your response:
    1.  Return ONLY a single, valid JSON object. Do not include any explanatory text, markdown, or anything outside the JSON structure.
    2.  The Python code inside the "code" field must be a valid JSON string. This means all special characters, especially newlines and double quotes, MUST be escaped (e.g., \\n for newlines, \\" for quotes).
    """
    user_prompt = f"""Previous tasks context:\n{pre_tasks_str}\n\n
    Generate Python code for the current tasks in the following structure:\n{json.dumps(curr_tasks, indent=2)}\n\n
    """

    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3-8b:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            logprobs=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "full_completion",
                    "schema": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},  # 子任务描述
                                "code": {"type": "string"}  # Python 代码
                            },
                            "required": ["task", "code"]
                        }
                    }
                }
            }
        )
        chat_completion = add_logprobs(completion)
        raw_output = chat_completion.value.choices[0].message.content
        logprobs = chat_completion.log_probs[0]
        # 解析 JSON
        try:
            task_dict = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}")
            return TaskOutput(tasks={}, transition_scores={})

        # 解析任务和代码
        completed_tasks = {}
        for k, v in task_dict.items():
            if isinstance(v, dict):
                completed_tasks[k] = SubTask(task=v.get("task", ""), code=v.get("code", ""))
            else:
                completed_tasks[k] = SubTask(task=str(v), code="")

        # 解析 logprobs 并转换成 SubLogpro
        transition_scores = {}
        for k, v in logprobs.items():
            if isinstance(v, dict) and "code" in v:
                transition_scores[k] = SubLogpro(task=float(np.exp(v["code"])))
            else:
                transition_scores[k] = SubLogpro(task=-np.inf)

        return TaskOutput(tasks=completed_tasks, transition_scores=transition_scores)

    except Exception as e:
        logging.error(f"An error occurred in generate_full_task_code: {e}")
        return TaskOutput(tasks={}, transition_scores={})


def write_jsonl(filename: str, data: List[Union[Dict, list, str]], append: bool = False):
    """
    Writes data to a JSON Lines file.

    Args:
        filename (str): The path to the file.
        data (List[Union[Dict, list, str]]): A list of data to write, where each element is typically a dictionary.
        append (bool): If True, appends to the end of the file. Otherwise, overwrites the file.
    """
    mode = "a" if append else "w"
    with open(filename, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class Node:
    """
    Represents a node in the Monte Carlo Tree Search (MCTS).

    Each node corresponds to a specific state in the code generation process, storing all relevant
    information such as parent-child relationships, visit counts, value, probability, and the
    associated task code.
    """
    # Use itertools.count() to generate a unique ID for each node instance
    id_iter = itertools.count()

    def __init__(self, logprob, overall_task, current_task, previous_task, parent, finised_step_id=0, value=0,
                 error='', label='default'):
        """
        Initializes a new Node.

        Args:
            logprob (float): The log probability of this node's state.
            overall_task (str): The description of the entire task.
            current_task (Dict): The task step and code represented by this node.
            previous_task (Dict): All completed task steps and code prior to this node.
            parent (Node or None): The parent node. The root node's parent is None.
            finised_step_id (int): The number of steps completed in the task sequence.
            value (float): The total reward obtained from this node (Q-value).
            error (str): Error message if this node's state contains an error.
            label (str): A label for the node, used for tracking its status (e.g., 'default', 'fix', 'dead').
        """
        self._children: List['Node'] = []  # List of child nodes
        self._parent: Optional['Node'] = parent  # Reference to the parent node
        self.visits: int = 1  # Number of times the node has been visited (N)
        self.runs: int = 0  # Number of times the node has been executed or attempted to be fixed
        self.finised_step_id: int = finised_step_id  # ID of the last completed task step
        self.value: float = value  # The value or total reward of the node (Q)
        self.prob: float = exp(logprob)  # The prior probability of the node (P), used in P-UCB calculation
        self.overall_task: str = overall_task  # The overall task description
        self.current_task: Dict = current_task  # The task and code for the current step
        self.previous_task: Dict = previous_task  # The tasks and code for previous steps
        self.id: int = next(self.id_iter)  # Unique ID for the node
        self.p_ucb: float = 0  # The latest calculated Polynomial Upper Confidence Bound (P-UCB) value
        self.error: str = error  # Stores error messages from child nodes

        # [NEW] Stores historical fix attempts for this node in the format {code: score}
        self.fix_attempts: Dict[str, float] = {}
        self.update: int = 0  # Records the number of times the node has been updated (e.g., task re-decomposition)
        self.label: str = label # A label for tracking the node's purpose or state

    def backprop(self, value: float):
        """
        Performs backpropagation of the reward up the tree.

        Args:
            value (float): The reward to propagate.
        """
        self.visits += 1
        self.value += value  # Alternatively, store sum_rewards and use value/visits as Q
        if self._parent is not None:
            self._parent.backprop(value)


class NodeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for serializing Node objects.

    This encoder removes direct references to parent and child nodes to avoid circular dependencies
    during serialization.
    """

    def default(self, obj):
        if isinstance(obj, Node):
            # Create a shallow copy of the object to avoid modifying the original
            cpy = copy.copy(obj)
            # Delete attributes that may cause circular references
            if hasattr(cpy, '_parent'):
                del cpy._parent
            if hasattr(cpy, '_children'):
                del cpy._children
            # Return the object's __dict__ representation
            return vars(cpy)
        # For other types, use the base class's default method
        return json.JSONEncoder.default(self, obj)


def collect_graph_data(node: Node) -> Tuple[Dict[int, Node], List[Tuple[int, int]]]:
    """
    Recursively traverses the tree from the root node to collect all nodes and edges for visualization.

    Args:
        node (Node): The root node to start traversal from.

    Returns:
        Tuple[Dict[int, Node], List[Tuple[int, int]]]:
            - A dictionary where keys are node IDs and values are the Node objects.
            - A list of all edges representing parent-child relationships as (parent_id, child_id).
    """
    nodes = {node.id: node}
    edges = []
    # Traverse all child nodes
    for child in node._children:
        # Add an edge from the current node to the child
        edges.append((node.id, child.id))
        # Recursively collect nodes and edges from the subtree
        child_nodes, child_edges = collect_graph_data(child)
        nodes.update(child_nodes)
        edges.extend(child_edges)
    return nodes, edges


def p_ucb_select(parent_node: Node, child_nodes: List[Node], c_base=10, c=4) -> Optional[Node]:
    """
    Selects a child node using the Polynomial Upper Confidence Bound (P-UCB) algorithm.

    P-UCB balances exploitation (choosing high-value nodes) and exploration (choosing less-visited nodes).

    Formula: p_ucb = Q(s,a) + β * P(s,a) * sqrt(log(N(s))) / (1 + N(s,a))
    where β = log((N(s) + c_base + 1) / c_base) + c

    Args:
        parent_node (Node): The current parent node.
        child_nodes (List[Node]): The list of child nodes to choose from.
        c_base (int): A constant in the P-UCB formula that influences the exploration weight.
        c (int): A constant in the P-UCB formula that influences the exploration weight.

    Returns:
        Optional[Node]: The child node with the highest P-UCB value. Returns None if the list is empty.
    """
    s_visits = parent_node.visits  # Total visits to the parent node N(s)
    # Calculate the exploration factor beta
    beta = log((s_visits + c_base + 1) / c_base) + c

    max_p_ucb = -inf
    max_node = None
    # Iterate through all child nodes to calculate their P-UCB values
    for node in child_nodes:
        # P-UCB calculation formula
        p_ucb = node.value + beta * node.prob * sqrt(log(s_visits)) / (1 + node.visits)

        # Print debugging information
        print('-----------------------------------------selecting node---------------------------------')
        print(f"Node ID {node.id}: P-UCB = {p_ucb}")
        # Store the latest P-UCB value for visualization
        node.p_ucb = p_ucb
        # Update the node with the highest P-UCB value
        if p_ucb > max_p_ucb:
            max_node = node
            max_p_ucb = p_ucb

    return max_node


def calculate_reward(
        pre_tasks: Dict[str, Dict[str, str]],
        completion: Dict[str, Dict[str, str]],
        exec_path: str,
        full_steps: int
) -> float:
    """
    Calculates a reward score based on the successful execution rate of task steps.

    This function first executes the code of all `pre_tasks` to establish a context, then
    executes the code for each step in the `completion` dictionary sequentially. The reward is
    based on the proportion of successfully executed steps in `completion`. If all steps
    succeed and the total number of tasks is reached, it returns a full score of 1.0.

    Args:
        pre_tasks (Dict): A dictionary containing previously completed steps.
                          Format: {'1': {'task': ..., 'code': ...}, ...}
        completion (Dict): A dictionary containing new steps to be evaluated.
                           Format: {'3': {'task': ..., 'code': ...}, ...}
        exec_path (str): The working directory for code execution.
        full_steps (int): The total number of steps required to complete the entire task.

    Returns:
        float: A reward score ranging from 0.0 to 1.0.
    """
    # If completion is empty, there are no new steps to evaluate, so the reward is 0
    if not completion:
        return 0.0

    # 1. Prepare the execution environment and base code context
    # Set the working directory and handle potential exceptions
    work_path_setup = f"import os\ntry:\n    os.chdir('{exec_path}')\nexcept Exception:\n    pass\n"

    # Concatenate the code of all pre_tasks to form the initial context
    pre_tasks_code = ""
    # Ensure pre_tasks are concatenated in numerical order
    sorted_pre_keys = sorted(pre_tasks.keys(), key=int)
    for key in sorted_pre_keys:
        pre_tasks_code += pre_tasks[key].get('code', '') + '\n'

    # 2. Execute the code in completion step-by-step and score
    successful_steps = 0
    cumulative_code = pre_tasks_code

    # Ensure the steps in completion are executed in numerical order
    sorted_completion_keys = sorted(completion.keys(), key=int)

    for key in sorted_completion_keys:
        step_code = completion[key].get('code', '')
        # Append the current step's code to the cumulative code
        current_execution_code = cumulative_code + step_code + '\n'
        current_execution_code = autopep8.fix_code(current_execution_code)

        # Execute the code in a monitored process
        print('---------starting reward calculation by executing the full code for step', key, '---------')
        error_message = monitor_process(
            work_path_setup + current_execution_code,
            MAX_EXECUTION_TIME,
            MAX_MEMORY_USAGE
        )
        print('---------execution finished---------')

        # Check if the execution was successful
        if error_message is None:
            # Current step succeeded, increment score, and update cumulative code for the next step
            successful_steps += 1
            cumulative_code = current_execution_code
        elif ('DeprecationWarning' in error_message) or ('ERROR' not in error_message) and (
                'Error' not in error_message) and ('Exception' not in error_message):
            # Some non-fatal outputs might also be considered successful
            successful_steps += 1
            cumulative_code = current_execution_code
        else:
            # Stop immediately if a step fails, subsequent steps will not be executed
            print(f"--- Execution failed at step '{key}' and finished reward calculation ---")
            print(f"Code:\n{step_code.strip()}")
            print(f"Error: {error_message}")
            print('---------stop executing further steps---------')
            break

    # 3. Calculate the final reward
    # Reward = number of successfully executed steps / total number of attempted steps
    reward = successful_steps / len(completion)

    # 4. Check if the entire task has been fully completed
    # Conditions: (1) all steps in completion succeeded (2) the number of the last step >= total steps
    all_completion_steps_succeeded = (successful_steps == len(completion))
    last_step_number = int(sorted_completion_keys[-1]) if sorted_completion_keys else -1

    if all_completion_steps_succeeded and last_step_number >= full_steps:
        print('************************** Full task completed successfully! *******************************')
        return 1.0  # Grant full reward

    return reward


def check_error_nodes(
        pre_tasks: Dict[str, Dict[str, str]],
        curr_task: Dict[str, Dict[str, str]],
        exec_path: str
) -> Tuple[str, float]:
    """
    Checks for errors in the current task's code and calculates the success rate (score) of the code itself.

    Args:
        pre_tasks (Dict): A dictionary of previously completed tasks.
        curr_task (Dict): A dictionary containing the single current task to check.
        exec_path (str): The working directory for code execution.

    Returns:
        Tuple[str, float]:
            - A string containing any found error messages.
            - A score from 0.0 to 1.0 indicating the percentage of code lines that executed successfully.
    """
    # 1. Extract the current code snippet
    if not isinstance(curr_task, dict) or len(curr_task) != 1:
        return "Input error: 'curr_task' must be a dictionary with a single entry.", 0.0

    step_key = list(curr_task.keys())[0]
    new_code_snippet = curr_task[step_key].get('code', '').strip()

    if not new_code_snippet:
        return "", 0.0  # No code to check

    # 2. Perform a syntax check
    try:
        tree = ast.parse(new_code_snippet)
    except SyntaxError as e:
        return f"SyntaxError: {e}", 0.0

    # 3. Prepare preceding code and execution environment
    pre_code = ""
    sorted_pre_keys = sorted(pre_tasks.keys(), key=int)
    for key in sorted_pre_keys:
        pre_code += pre_tasks[key].get('code', '') + '\n'
    work_path_setup = f"import os\ntry:\n    os.chdir('{exec_path}')\nexcept Exception:\n    pass\n"

    # --- Score Calculation Logic ---
    line_to_nodes = {}
    for node in ast.walk(tree):
        if hasattr(node, 'lineno'):
            line_to_nodes.setdefault(node.lineno, []).append(node)

    unique_lines = []
    new_context = ""
    for line_key, line_values in sorted(line_to_nodes.items()):
        line = max([ast.unparse(v) for v in line_values], key=len)
        if line not in new_context:
            new_context += line
            unique_lines.append(line + "\n")

    code_len = len(unique_lines)
    if code_len == 0:
        return "", 0.0

    success_len = 0
    test_code_context = pre_code
    error_message = ""

    for code_i in unique_lines:
        temp_code = test_code_context + "\n" + code_i
        temp_code = autopep8.fix_code(temp_code)
        return_back = monitor_process(
            work_path_setup + temp_code,
            MAX_EXECUTION_TIME,
            MAX_MEMORY_USAGE
        )
        if return_back is None or (
                'ERROR' not in return_back and 'Error' not in return_back and 'Exception' not in return_back):
            success_len += 1
            test_code_context += code_i  # Update context
        else:
            error_message = return_back
            break  # Stop on first error

    score = success_len / code_len if code_len > 0 else 0.0
    # --- End of Score Calculation ---

    if not error_message:
        return "", score

    # 4. Analyze the error message
    found_errors = []
    if 'NameError' in error_message:
        match = re.search(r"name '([^']*)' is not defined", error_message)
        if match:
            undefined_name = match.group(1)
            found_errors.append(f"Undefined variable '{undefined_name}'.")

    elif 'AttributeError' in error_message and jedi is not None:
        match = re.search(r"(?:module '|\w+' object|'(\w+)') has no attribute '(\w+)'", error_message)
        if match:
            obj_name, attr_name = match.groups()[-2:]
            found_errors.append(f"Object '{obj_name}' has no attribute '{attr_name}'.")
            try:
                script = jedi.Script(pre_code + new_code_snippet)
                lines = (pre_code + new_code_snippet).splitlines()
                last_line = len(lines)
                last_col = len(lines[-1]) if lines else 0
                completions = script.complete(line=last_line, column=last_col)
                obj_completions = [c.name for c in completions if c.name.startswith(attr_name[:2])]
                if obj_completions:
                    found_errors.append(f"Did you mean: {', '.join(obj_completions[:5])}?")
            except Exception as e:
                found_errors.append(f"Jedi analysis failed: {e}")

    elif ('DeprecationWarning' in error_message) or ('ERROR' in error_message) or ('Error' in error_message) or (
            'EEException' in error_message):
        found_errors.append(f"Runtime error: {error_message}")

    return "\n".join(found_errors), score


def check_child_nodes(
        pre_tasks: Dict[str, Dict[str, str]],
        curr_task: Dict[str, Dict[str, str]],
        exec_path: str
) -> str:
    """
    Checks the current task's code for various potential errors, given the context of previous tasks.

    This function performs both syntax and runtime checks, and uses Jedi to provide additional
    information for specific errors like AttributeError. All found errors are combined into a
    single string and returned.

    Args:
        pre_tasks (Dict): A dictionary containing all previous steps.
        curr_task (Dict): A single-element dictionary containing the current step to be checked.
        exec_path (str): The working directory for code execution.

    Returns:
        str: A string containing all error messages. Returns an empty string if the code is valid.
    """
    # 1. Safely extract the code from the input dictionary
    if not isinstance(curr_task, dict) or len(curr_task) != 1:
        return "Input error: 'curr_task' must be a dictionary with a single entry."

    step_key = list(curr_task.keys())[0]
    new_code_snippet = curr_task[step_key].get('code', '').strip()

    if not new_code_snippet:
        return ""  # Empty code is considered valid

    # 2. Syntax Check - The first gate
    try:
        ast.parse(new_code_snippet)
    except SyntaxError as e:
        # Syntax errors are fatal, return immediately
        return f"SyntaxError: {e}"

    # 3. Prepare the full code context for runtime checks and Jedi analysis
    pre_code = ""
    sorted_pre_keys = sorted(pre_tasks.keys(), key=int)
    for key in sorted_pre_keys:
        pre_code += pre_tasks[key].get('code', '') + '\n'

    full_code = pre_code + new_code_snippet
    full_code = autopep8.fix_code(full_code)
    work_path_setup = f"import os\ntry:\n    os.chdir('{exec_path}')\nexcept Exception:\n    pass\n"

    print('---------starting to execute the full code for runtime check---------')
    # 4. Runtime Check
    error_message = monitor_process(
        work_path_setup + full_code,
        MAX_EXECUTION_TIME,
        MAX_MEMORY_USAGE
    )
    print('---------execution finished---------')

    if not error_message:
        return ""  # No errors, return an empty string

    # 5. In-depth analysis of the captured runtime error
    found_errors = []
    # --- NameError Analysis ---
    if 'NameError' in error_message:
        match = re.search(r"name '([^']*)' is not defined", error_message)
        if match:
            undefined_name = match.group(1)
            found_errors.append(f"  - Detailed Analysis: Undefined variable name '{undefined_name}' detected. "
                                f"Please check if the variable has been declared or if there is a typo.")

    # --- AttributeError Analysis (with Jedi) ---
    elif 'AttributeError' in error_message and jedi is not None:
        match = re.search(r"(?:module '|\w+' object|'(\w+)') has no attribute '(\w+)'", error_message)
        if match:
            obj_name, attr_name = match.groups()[-2:]
            found_errors.append(f"  - Detailed Analysis: Object '{obj_name}' has no attribute named '{attr_name}'.")
            try:
                script = jedi.Script(full_code)
                lines = full_code.splitlines()
                last_line = len(lines)
                last_col = len(lines[-1]) if lines else 0
                completions = script.complete(line=last_line, column=last_col)
                obj_completions = [c.name for c in completions if c.name.startswith(attr_name[:2])]
                if obj_completions:
                    suggestions = ", ".join(obj_completions[:5])
                    found_errors.append(f"  - Jedi Suggestion: Did you mean to type one of these on '{obj_name}': {suggestions}?")
                else:
                    found_errors.append(f"  - Jedi Suggestion: Could not find any attributes related to '{attr_name}' on '{obj_name}'.")
            except Exception as e:
                found_errors.append(f"  - Error during Jedi analysis: {e}")

    elif ('DeprecationWarning' in error_message) or ('ERROR' in error_message) or ('Error' in error_message) or (
            'EEException' in error_message):
        found_errors.append(f"Runtime error: {error_message}")

    # 6. Combine all collected error messages into a single string
    return "\n".join(found_errors)


# ==============================================================================
#  DATA TRANSFORMATION AND LOGGING
# ==============================================================================
def transform_task_data(input_data: Dict, library_list: List[str]) -> Dict:
    """
    Transforms an input task dictionary into a specific format for the MCTS process.

    Args:
        input_data (Dict): The original input dictionary containing a "tasks" key.
        library_list (List[str]): A list of libraries to be assigned to each task.

    Returns:
        Dict: A transformed dictionary containing task information, keyed by task ID.
    """
    tasks_list = {}
    previous_task = {}

    # Sort task IDs to ensure they are processed in order
    sorted_task_ids = sorted(input_data['tasks'].keys(), key=int)

    for task_id_str in sorted_task_ids:
        task_id = str(task_id_str)  # Ensure it is a string
        task_description = input_data['tasks'][task_id]['task']
        task_code = input_data['tasks'][task_id].get('code', '')
        task_score = input_data['transition_scores'][task_id].get('task', 0)

        # Construct the pre_task for the current step
        current_pre_task = copy.deepcopy(previous_task)

        transformed_task = {
            'task_id': task_id,
            'library': library_list,
            'curr_task': {'task': task_description, 'code': task_code, 'score': task_score, 'ori_id': task_id},
            'pre_task': current_pre_task,
        }
        tasks_list[task_id] = transformed_task

        # Update previous_task for the next iteration
        previous_task[task_id] = {'task': task_description, 'code': task_code, 'score': task_score,
                                  'ori_id': task_id}

    return tasks_list


def taskoutput_to_dict(task_output: TaskOutput) -> dict:
    """
    Converts a TaskOutput object to a dictionary with 'task', 'code', and 'score' for each task.

    Args:
        task_output (TaskOutput): The TaskOutput object to convert.

    Returns:
        dict: A dictionary representation of the tasks.
    """
    result = {}
    sorted_task_ids = sorted(task_output.tasks.keys(), key=lambda x: int(x))  # Sort numerically
    for task_id_str in sorted_task_ids:
        task_id = str(task_id_str)  # Ensure it is a string
        subtask = task_output.tasks[task_id]
        task_score = task_output.transition_scores.get(task_id, SubLogpro(task=0)).task
        result[task_id] = {
            'task': subtask.task,
            'code': subtask.code,
            'score': task_score,
            'ori_id': task_id
        }
    return result


def insert_tasks_to_dict(
        tasks: Dict[str, Dict[str, Any]],
        on_process_tasks: Dict[str, Dict[str, Any]],
        insert_after_id: Union[int, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Inserts a set of new tasks into an existing task processing queue after a specified position.
    Preserves the original structure of curr_task (task/code/score).

    Args:
        tasks (Dict): The new tasks to insert, where each value contains a curr_task.
        on_process_tasks (Dict): The current task queue.
        insert_after_id (Union[int, str]): The task ID after which the new tasks should be inserted.

    Returns:
        Dict: The re-numbered complete task queue after insertion.
    """
    insert_after_id = str(insert_after_id)

    # 1. Convert on_process_tasks to a list sorted by numerical order of keys
    ordered_keys = sorted(on_process_tasks.keys(), key=int)
    task_list = [on_process_tasks[k] for k in ordered_keys]

    # 2. Locate the insertion position
    if insert_after_id not in on_process_tasks:
        raise KeyError(f"insert_after_id '{insert_after_id}' not found in on_process_tasks")
    pos = ordered_keys.index(insert_after_id)

    # 3. Generate a list of the new tasks to be inserted
    insert_list = [t for t in tasks.values()]

    # 4. Concatenate the lists at the specified position
    new_list = task_list[:pos + 1] + insert_list + task_list[pos + 1:]

    # 5. Re-number and convert back to a dictionary
    new_dict = {}
    for i, item in enumerate(new_list):
        item["task_id"] = str(i)
        new_dict[str(i)] = item

    return new_dict


def record_final_step_decision(
        filename: str,
        decision_node: 'Node',
):
    """
    Records the final decision for a confirmed task step to a JSON Lines file.

    This function extracts the current task information and key MCTS metadata
    from the provided decision node and saves it as a single JSON object on a new line.

    Args:
        filename (str): The path to the JSON Lines file where the record will be appended.
        decision_node (Node): The node representing the final, chosen solution for the current step.
    """
    # Create a dictionary containing the final decision and relevant metadata.
    record = {
        'final_task_step': decision_node.current_task,
        # Extract metadata from the node for traceability.
        'node_id': decision_node.id,
        'node_value': decision_node.value,
        'node_visits': decision_node.visits
    }

    # Append the record as a new line to the specified log file.
    write_jsonl(filename, [record], append=True)


def record_rollout_state(
        rollout_key: str,
        root: Node,
        selection_path: List[int],
        action: str,
        action_details: Optional[Dict] = None,
        save_filename: Optional[str] = None
):
    """
    Captures a snapshot of the current MCTS tree and appends it as a record to a log file.

    Args:
        rollout_key (str): A unique identifier for the rollout.
        root (Node): The root node of the MCTS tree.
        selection_path (List[int]): The path of node IDs selected during this rollout.
        action (str): The action taken during this rollout (e.g., 'fix', 'expansion').
        action_details (Optional[Dict]): Additional details about the action.
        save_filename (Optional[str]): The filename to save the log to.
    """
    if not save_filename:
        print("Warning: No save_filename provided to record_rollout_state. State not saved.")
        return

    # 1. Collect the current snapshot of the graph
    all_nodes, all_edges = collect_graph_data(root)
    # [KEY CHANGE]: No longer manually call make_serializable. Use the raw all_nodes dictionary.
    # The NodeEncoder will handle them in the json.dumps step below.

    # 2. Construct the data to be written to the file
    rollout_data = {
        "action": action,
        "selected_nodes_path": selection_path,
        # make_serializable might still be necessary here if action_details contains other complex objects
        "action_details": make_serializable(action_details) if action_details else {},
        "tree_snapshot": {"nodes": all_nodes, "edges": all_edges}  # <-- Pass all_nodes directly
    }

    # 3. Append this record to the .jsonl file
    save_rollout_snapshot_to_jsonl(save_filename, rollout_key, rollout_data)


def save_rollout_snapshot_to_jsonl(filename: str, rollout_key: str, data: Dict):
    """
    Appends the data for a single rollout as one line to a JSON Lines file, always using NodeEncoder.

    Args:
        filename (str): The name of the file to save to.
        rollout_key (str): The unique key for the rollout.
        data (Dict): The data to be saved.
    """
    line_data = {"rollout_key": rollout_key, **data}
    try:
        with open(filename, "a", encoding="utf-8") as f:
            # Directly and exclusively use NodeEncoder for serialization
            f.write(json.dumps(line_data, cls=NodeEncoder, ensure_ascii=False, indent=None) + "\n")
    except TypeError as e:
        # If an error still occurs, we will now see a clear error message instead of it being silenced
        print(f"!!! A critical error occurred during log serialization: {e}")
        # Consider writing an error marker instead of malformed data
        error_line = {"rollout_key": rollout_key, "error": f"Serialization failed: {e}"}
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_line) + "\n")

