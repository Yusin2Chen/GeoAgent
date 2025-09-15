#!pip install openai
#!pip install jedi
#!pip install timeout-decorator
#!pip install structured-logprobs
#!pip install earthengine-api
#!pip install geemap
#!pip install autopep8
#!pip install pillow networkx matplotlib
#!pip install pydot
import os, re
import jedi
import autopep8
import time
from multi_process import monitor_process, make_serializable, is_serializable
from math import exp, log, inf, sqrt
import ast
import itertools, copy
import json
from openai import OpenAI
from structured_logprobs.main import add_logprobs
from typing import List, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel, RootModel
from utils import decompose_task, generate_task_codes, llm_task_update, generate_full_task_code, TaskOutput, SubTask, SubLogpro,\
TaskCodeWithCandidates, CandidateCode, re_decompose_task

# --- OpenAI Client Configuration ---
# IMPORTANT: Replace the placeholder with your actual OpenRouter.ai API key.
# You can get a key from https://openrouter.ai/
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key='sk-or-v1-0fbb1ae0xxxxxxxxxxxxxxxxxxxxxxxxxxx')


def write_jsonl(filename: str, data: List[Union[Dict, list, str]], append: bool = False):
    """
    Writes data to a JSON Lines file.

    Args:
        filename: The file path.
        data: A list of data to write, where each element is typically a dict.
        append: Whether to append to the end of the file. If False, the file will be overwritten.
    """
    mode = "a" if append else "w"
    with open(filename, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

class Node:
    """
    A node in the Monte Carlo Tree Search (MCTS) tree.

    Each node represents a specific code generation state and stores all information
    related to that state, including its parent-child relationships, visit counts,
    value, probability, and corresponding task code.
    """
    # Use itertools.count() to generate a unique ID for each node instance
    id_iter = itertools.count()

    def __init__(self, logprob, overall_task, current_task, previous_task, parent, finised_step_id=0, value=0, error='', label='default'):
        """
        Initializes a node.

        Args:
            logprob (float): The log probability of this node's state.
            overall_task (str): The description of the overall task.
            current_task (Dict): The task step and its code represented by the current node.
            previous_task (Dict): All completed task steps and their code prior to this node.
            parent (Node or None): The parent node. The parent of the root node is None.
            finised_step_id (int): The number of completed steps in the task sequence.
            value (float): The total reward obtainable from this node (Q-value).
            error (str): If an error exists in this node's state, record the error message.
        """
        self._children = []  # List of child nodes
        self._parent = parent  # Reference to the parent node
        self.visits = 1  # Number of times the node has been visited (N)
        self.runs = 0  # Number of times the node has been executed or attempted to be fixed
        self.finised_step_id = finised_step_id  # ID of the completed task step
        self.value = value  # The value or total reward of the node (Q)
        self.prob = exp(logprob)  # The prior probability of the node (P), used for P-UCB calculation
        self.overall_task = overall_task  # Overall task description
        self.current_task = current_task  # Task and code for the current step
        self.previous_task = previous_task  # Task and code for previous steps
        self.id = next(self.id_iter)  # Unique ID for the node
        self.p_ucb = 0  # The most recently calculated P-UCB (Polynomial Upper Confidence Bound) value
        self.error = error  # Records error messages from child nodes

        # [NEW] Store historical fix attempts for this node, in the format {code: score}
        self.fix_attempts = {}
        self.update = 0  # Records the number of times the node has been updated (e.g., task re-decomposition)
        # Not sure what this is for
        self.label = label

    def backprop(self, value):
        self.visits += 1
        self.value += value  # Alternatively, store sum_rewards and use value/visits as Q
        if self._parent is not None:
            self._parent.backprop(value)


class NodeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for serializing Node objects.

    During serialization, direct references to parent and child nodes are removed
    to avoid circular dependencies.
    """
    def default(self, obj):
        if isinstance(obj, Node):
            # Create a shallow copy of the object to avoid modifying the original
            cpy = copy.copy(obj)
            # Delete attributes that could cause circular references
            del cpy._parent
            del cpy._children
            # Return the object's __dict__ representation
            return vars(cpy)
        # For other types, use the base class's default method
        return json.JSONEncoder.default(self, obj)


def collect_graph_data(node: Node) -> Tuple[Dict[int, Node], List[Tuple[int, int]]]:
    """
    Recursively traverses from the root node to collect all nodes and edges
    in the tree for visualization.

    Args:
        node (Node): The root node to start traversal from.

    Returns:
        Tuple[Dict[int, Node], List[Tuple[int, int]]]:
            - A dictionary where keys are node IDs and values are node objects.
            - A list of all edges representing parent-child relationships (parent_id, child_id).
    """
    nodes = {node.id: node}
    edges = []
    # Traverse all child nodes
    for child in node._children:
        # Add an edge from the current node to the child node
        edges.append((node.id, child.id))
        # Recursively collect nodes and edges from the subtree
        child_nodes, child_edges = collect_graph_data(child)
        nodes.update(child_nodes)
        edges.extend(child_edges)
    return nodes, edges


def p_ucb_select(parent_node: Node, child_nodes: List[Node], c_base=10, c=4) -> Optional[Node]:
    """
    Uses the P-UCB (Polynomial Upper Confidence Bound) algorithm to select a child node.

    The P-UCB algorithm balances exploitation (choosing high-value nodes) and
    exploration (choosing less-visited nodes).

    Formula: p_ucb = Q(s,a) + β * P(s,a) * sqrt(log(N(s))) / (1 + N(s,a))
    where β = log((N(s) + c_base + 1) / c_base) + c

    Args:
        parent_node (Node): The current parent node.
        child_nodes (List[Node]): The list of child nodes to choose from.
        c_base (int): A constant in the P-UCB formula that affects the exploration weight.
        c (int): A constant in the P-UCB formula that affects the exploration weight.

    Returns:
        Node or None: The child node with the highest P-UCB value. Returns None if the list is empty.
    """
    s_visits = parent_node.visits  # Total visit count of the parent node N(s)
    # Calculate the exploration factor beta
    beta = log((s_visits + c_base + 1) / c_base) + c
    #print(s_visits, beta)

    max_p_ucb = -inf
    max_node = None
    # Iterate through all child nodes to calculate their P-UCB values
    for node in child_nodes:
        # P-UCB calculation formula
        p_ucb = node.value + beta * node.prob * sqrt(log(s_visits)) / (1 + node.visits)
        #print(node.value, node.prob, s_visits, node.visits)

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

    This function first executes the code of all pre_tasks to establish a context,
    then executes the code for each step in the completion dictionary one by one.
    The reward is based on the proportion of successfully executed steps in completion.
    If all steps succeed and the total number of tasks is reached, it returns a full score of 1.0.

    Args:
        pre_tasks (Dict): A dictionary containing previously completed steps.
                          Format: {'1': {'task': ..., 'code': ...}, ...}
        completion (Dict): A dictionary containing new steps to be evaluated.
                           Format: {'3': {'task': ..., 'code': ...}, ...}
        exec_path (str): The working directory for code execution.
        full_steps (int): The total number of steps required to complete the entire task.

    Returns:
        float: A reward score, ranging from 0.0 to 1.0.
    """
    # If completion is empty, it means there are no new steps to evaluate, reward is 0
    if not completion:
        return 0.0

    # 1. Prepare the execution environment and base code context
    # Set the working directory and handle potential exceptions
    work_path_setup = f"import os\ntry:\n    os.chdir('{exec_path}')\nexcept Exception:\n    pass\n"

    # Concatenate the code of all pre_tasks as the initial context
    pre_tasks_code = ""
    # Ensure pre_tasks are concatenated in numerical order
    sorted_pre_keys = sorted(pre_tasks.keys(), key=int)
    for key in sorted_pre_keys:
        pre_tasks_code += pre_tasks[key].get('code', '') + '\n'

    # 2. Execute the code in completion step-by-step and score it
    successful_steps = 0
    cumulative_code = pre_tasks_code

    # Ensure the steps in completion are executed in numerical order
    sorted_completion_keys = sorted(completion.keys(), key=int)

    for key in sorted_completion_keys:
        step_code = completion[key].get('code', '')
        # Append the current step's code to the cumulative code
        current_execution_code = cumulative_code + step_code + '\n'
        current_execution_code = autopep8.fix_code(current_execution_code)

        # Execute the code using a monitored process
        print('---------starting reward calculation by executing the full code for step', key, '---------')
        error_message = monitor_process(
            work_path_setup + current_execution_code,
            MAX_EXECUTION_TIME,
            MAX_MEMORY_USAGE
        )
        print('---------execution finished---------')

        # Determine if the execution was successful
        if error_message is None:
            # Current step succeeded, score it, and update cumulative code for the next step
            successful_steps += 1
            cumulative_code = current_execution_code
        elif ('DeprecationWarning' in error_message) or ('ERROR' not in error_message) and ('Error' not in error_message) and ('Exception' not in error_message):
            # Some non-fatal outputs might also be considered successful
            successful_steps += 1
            cumulative_code = current_execution_code
        else:
            # If a step fails, stop immediately; subsequent steps are not executed
            print(f"--- Execution failed at step '{key}' and finished reward calculation ---")
            print(f"Code:\n{step_code.strip()}")
            print(f"Error: {error_message}")
            print('---------stop executing further steps---------')
            break

    # 3. Calculate the final reward
    # Reward = number of successful steps / total number of attempted steps
    reward = successful_steps / len(completion)

    # 4. Determine if the entire task has been fully completed
    # Conditions: (1) all steps in completion succeeded (2) the number of the last step >= total steps
    all_completion_steps_succeeded = (successful_steps == len(completion))
    last_step_number = int(sorted_completion_keys[-1]) if sorted_completion_keys else -1

    if all_completion_steps_succeeded and last_step_number >= full_steps:
        print('************************** Full task completed successfully! *******************************')
        return 1.0  # Grant a full score reward

    return reward


def get_best_program(program_dict: Dict[str, float]) -> Tuple[Optional[str], float]:
    """
    Selects the program with the highest reward from a dictionary of programs and their corresponding rewards.

    Args:
        program_dict (Dict[str, float]): A dictionary where keys are program code and values are their reward scores.

    Returns:
        Tuple[Optional[str], float]:
            - The program code with the highest reward. None if the dictionary is empty.
            - The corresponding highest reward score.
    """
    max_reward = -inf
    best_program = None
    for program, reward in program_dict.items():
        #print('-----------------------select best completion-----------------------')
        #print(program, reward)
        if reward > max_reward:
            best_program = program
            max_reward = reward
    return best_program, max_reward


def check_error_nodes(
        pre_tasks: Dict[str, Dict[str, str]],
        curr_task: Dict[str, Dict[str, str]],
        exec_path: str
) -> Tuple[str, float]:
    """
    Checks for errors in the current task's code and calculates the success rate (score) of the code itself.
    """

    found_errors = []

    # 1. Extract the current code
    if not isinstance(curr_task, dict) or len(curr_task) != 1:
        return "Input error: 'curr_task' must be a dictionary with a single entry.", 0.0

    step_key = list(curr_task.keys())[0]
    new_code_snippet = curr_task[step_key].get('code', '').strip()

    if not new_code_snippet:
        return "", 0.0

    # 2. Syntax check
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

    # ----------- score calculation logic -----------
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
        if return_back is None or ('ERROR' not in return_back and 'Error' not in return_back and 'Exception' not in return_back):
            success_len += 1
            test_code_context += code_i  # Update the context
        else:
            error_message = return_back
            break  # Stop on the first error

    score = success_len / code_len if code_len > 0 else 0.0
    # ----------- end of score calculation -----------

    if not error_message:
        return "", score

    # 4. Analyze the error
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
                found_errors.append(f"Jedi analysis error: {e}")


    elif ('DeprecationWarning' in error_message) or ('ERROR' in error_message) or ('Error' in error_message) or ('EEException' in error_message):
        found_errors.append(f"Runtime error: {error_message}")

    return "\n".join(found_errors), score



def check_child_nodes(
        pre_tasks: Dict[str, Dict[str, str]],
        curr_task: Dict[str, Dict[str, str]],
        exec_path: str
) -> str:
    """
    Checks the current task's code for various potential errors, given the context of previous tasks.

    This function performs syntax and runtime checks, and uses Jedi to provide
    additional information for specific errors (like AttributeError). All found
    errors are combined into a single string and returned.

    Args:
        pre_tasks (Dict): A dictionary containing all previous steps.
        curr_task (Dict): A single-element dictionary containing the current step to be checked.
        exec_path (str): The working directory for code execution.

    Returns:
        str: A string containing all error information. Returns an empty string if the code is valid.
    """
    found_errors = []

    # 1. Safely extract the code from the input dictionary
    if not isinstance(curr_task, dict) or len(curr_task) != 1:
        return "Input error: 'curr_task' must be a dictionary with a single entry."

    step_key = list(curr_task.keys())[0]
    new_code_snippet = curr_task[step_key].get('code', '').strip()

    if not new_code_snippet:
        return ""  # Empty code is considered valid

    # 2. Syntax Check - this is the first gate
    try:
        ast.parse(new_code_snippet)
    except SyntaxError as e:
        # Syntax errors are critical, return immediately
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

    #print(repr(error_message))

    if not error_message:
        return ""  # No errors, return an empty string

    # 5. In-depth analysis of the captured runtime error
    # --- NameError Analysis ---
    if 'NameError' in error_message:
        match = re.search(r"name '([^']*)' is not defined", error_message)
        if match:
            undefined_name = match.group(1)
            found_errors.append(f"  - Detailed analysis: Detected undefined variable name '{undefined_name}'. Please check if the variable is declared or spelled correctly.")

    # --- AttributeError Analysis (using Jedi) ---
    elif 'AttributeError' in error_message and jedi is not None:
        match = re.search(r"(?:module '|\w+' object|'(\w+)') has no attribute '(\w+)'", error_message)
        if match:
            obj_name, attr_name = match.groups()[-2:]
            found_errors.append(f"  - Detailed analysis: Object '{obj_name}' has no attribute named '{attr_name}'.")
            try:
                script = jedi.Script(full_code)
                lines = full_code.splitlines()
                last_line = len(lines)
                last_col = len(lines[-1]) if lines else 0
                completions = script.complete(line=last_line, column=last_col)
                obj_completions = [c.name for c in completions if c.name.startswith(attr_name[:2])]
                if obj_completions:
                    suggestions = ", ".join(obj_completions[:5])
                    found_errors.append(f"  - Jedi suggestion: On '{obj_name}', did you mean to type: {suggestions}?")
                else:
                    found_errors.append(f"  - Jedi suggestion: Could not find attributes related to '{attr_name}' on '{obj_name}'.")
            except Exception as e:
                found_errors.append(f"  - Error during Jedi analysis: {e}")
    elif ('DeprecationWarning' in error_message) or ('ERROR' in error_message) or ('Error' in error_message) or ('EEException' in error_message):
        found_errors.append(f"Runtime error: {error_message}")

    # 6. Combine all collected error messages into a single string
    return "\n".join(found_errors)


def transform_task_data(input_data, library_list):
    """
    Transforms the input task dictionary into a specific format for the MCTS process.

    Args:
      input_data (dict): The original input dictionary containing a "tasks" key.
      library_list (list): A list of libraries to be assigned to each task.

    Returns:
      dict: A transformed dictionary containing task information, keyed by task ID.
    """
    tasks_list = {}
    previous_task = {}

    # Sort the task IDs to ensure they are processed in order
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

        # Update previous_task for the next loop
        previous_task[task_id] = {'task': task_description, 'code': task_code, 'score': task_score, 'ori_id': task_id}

    return tasks_list

def taskoutput_to_dict(task_output: TaskOutput) -> dict:
    """Convert TaskOutput to dict with 'task', 'code', 'score' for each task."""
    result = {}
    sorted_task_ids = sorted(task_output.tasks.keys(), key=lambda x: int(x))  # Sort by numerical order
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
        tasks: Dict[str, Dict[str, Any]],  # e.g., {"1":{"curr_task": {...}, ...}, "2":{...}}
        on_process_tasks: Dict[str, Dict[str, Any]],  # e.g., {"0":{...}, "1":{...}}
        insert_after_id: int | str  # Insert after this id
) -> Dict[str, Dict[str, Any]]:
    """
    Inserts a set of new tasks into an existing task processing queue after a specified position.
    Preserves the original structure of curr_task (task/code/score).

    Args:
        tasks (Dict): The dictionary of new tasks to insert, each value containing a curr_task.
        on_process_tasks (Dict): The current task queue.
        insert_after_id (int | str): The task ID after which the new tasks will be inserted.

    Returns:
        Dict: The complete task queue, renumbered after inserting the new tasks.
    """
    insert_after_id = str(insert_after_id)

    # 1. Convert on_process_tasks to a list sorted by numerical order
    ordered_keys = sorted(on_process_tasks.keys(), key=int)
    task_list = [on_process_tasks[k] for k in ordered_keys]

    # 2. Locate the insertion position
    if insert_after_id not in on_process_tasks:
        raise KeyError(f"insert_after_id '{insert_after_id}' is not in on_process_tasks")
    pos = ordered_keys.index(insert_after_id)

    # 3. Generate a list of new tasks to be inserted
    insert_list = [t for t in tasks.values()]
    # 4. Concatenate the lists at the specified position
    new_list = task_list[:pos + 1] + insert_list + task_list[pos + 1:]

    # 5. Renumber and restore to a dictionary
    new_dict = {}
    for i, item in enumerate(new_list):
        item["task_id"] = str(i)
        new_dict[str(i)] = item

    return new_dict


def record_final_step_decision(
    filename: str,
    decision_node: Node,
):
    """
    Records the final decision for a confirmed task step.
    """
    # Extract the information of the current step from the decision node
    item = dict(
        curr_task=decision_node.current_task,
        # Get some metadata from the node
        node_id=decision_node.id,
        node_value=decision_node.value,
        node_visits=decision_node.visits
    )
    write_jsonl(filename, [item], append=True)


# New version - please use this one
def record_rollout_state(
        rollout_key: str, root: Node, selection_path: List[int],
        action: str,
        action_details: Optional[Dict] = None,
        save_filename: str = None
):
    """
    Captures a snapshot of the current MCTS tree and appends it as a record to the log file.
    """
    if not save_filename:
        print("Warning: save_filename not provided to record_rollout_state. State not saved.")
        return

    # 1. Collect the current snapshot of the graph
    all_nodes, all_edges = collect_graph_data(root)
    # [KEY CHANGE]: No longer manually calling make_serializable. The raw all_nodes dictionary is used directly.
    # NodeEncoder will handle them in the json.dumps step below.

    # 2. Construct the data to be written to the file
    rollout_data = {
        "action": action,
        "selected_nodes_path": selection_path,
        # make_serializable might be necessary here if action_details contains other complex objects
        "action_details": make_serializable(action_details) if action_details else {},
        "tree_snapshot": {"nodes": all_nodes, "edges": all_edges} # <--- Passing all_nodes directly
    }

    # 3. Append this record to the .jsonl file
    save_rollout_snapshot_to_jsonl(save_filename, rollout_key, rollout_data)


# New version - please use this one
def save_rollout_snapshot_to_jsonl(filename: str, rollout_key: str, data: Dict):
    """Appends the data of a single rollout as one line to a JSON Lines file, always using NodeEncoder."""
    line_data = {"rollout_key": rollout_key, **data}
    try:
        with open(filename, "a", encoding="utf-8") as f:
            # Directly and exclusively use NodeEncoder to handle serialization
            f.write(json.dumps(line_data, cls=NodeEncoder, ensure_ascii=False, indent=None) + "\n")
    except TypeError as e:
        # If an error still occurs, we will now see a clear error message instead of it being silently handled
        print(f"!!! Serious error during log serialization: {e}")
        # Consider writing an error marker here instead of malformed data
        error_line = {"rollout_key": rollout_key, "error": f"Serialization failed: {e}"}
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_line) + "\n")


if __name__ == "__main__":
    task_path = 'D:\\pythonCode\\pythonCode\\MCTS_Agent\\workspace\\GEE_0004.json'
    output_doc = 'D:\\pythonCode\\pythonCode\\MCTS_Agent\\workspace\\GEE_0004_multi.json'
    exec_path = './workspace'
    library = ['numpy', 'geemap', 'ee']
    task_name = os.path.basename(task_path).split('.')[0]
    # [NEW] Define clearly separated log files
    final_solution_log_filename = f"{task_name}_solution_steps.jsonl"
    graph_log_filename = f"graph_{task_name}_rollout_log.jsonl"  # Unchanged
    # [NEW] Clear old final result logs before starting
    if os.path.exists(final_solution_log_filename):
        os.remove(final_solution_log_filename)
    if os.path.exists(graph_log_filename):
        os.remove(graph_log_filename)

    # MCTS Hyperparameters
    MAX_EXECUTION_TIME = 10  # seconds
    MAX_MEMORY_USAGE = 512  # MB
    max_rollouts = 20  # Maximum number of rollouts
    max_runs_per_node = 2  # Maximum number of attempts/fixes per node
    top_k = 3  # Number of code candidates to generate during expansion

    # --- b. Load and decompose the initial task ---
    tasks = json.load(open(task_path))
    task_list = [v['task'] for k, v in tasks.items()]
    complete_task = '\n'.join(task_list)
    # Call LLM for task decomposition
    decomposed_tasks = decompose_task(complete_task, pre_task='let\'s print hello to start.')
    if not decomposed_tasks.tasks:
        print('----------Task Decomposition Failed----------')
    ##decomposed_tasks = TaskOutput(tasks={'0': SubTask(task='print hello', code=''), '1': SubTask(task="initialize Earth Engine and define region of interest using asset path 'projects/ee-ecresener5/assets/jokkmokk'", code=''), '2': SubTask(task='define function to mask clouds and shadows using the SCL band', code=''), '3': SubTask(task='define function to calculate NDVI using B8 and B4 bands', code=''), '4': SubTask(task='create cloud-masked median composite for 2019 summer period (June 1 to August 31) over ROI, calculate NDVI, and rename bands with year', code=''), '5': SubTask(task='create cloud-masked median composite for 2020 summer period (June 1 to August 31) over ROI, calculate NDVI, and rename bands with year', code=''), '6': SubTask(task='create cloud-masked median composite for 2021 summer period (June 1 to August 31) over ROI, calculate NDVI, and rename bands with year', code=''), '7': SubTask(task='combine the three annual composite images into a single multi-band image', code=''), '8': SubTask(task='initialize interactive map and display the ROI and a true-color view of the 2021 composite', code=''), '9': SubTask(task="export the combined multi-band composite image to Google Drive as 'Annual_Summer_Composites_Jokkmokk' with 10-meter resolution and cloud-optimized GeoTIFF format", code='')}, transition_scores={'0': SubLogpro(task=-0.4159837565239286), '1': SubLogpro(task=-0.7280267480525993), '2': SubLogpro(task=-2.27854277176084), '3': SubLogpro(task=-1.9314369674923455), '4': SubLogpro(task=-9.400597509737622), '5': SubLogpro(task=-8.100786033280201), '6': SubLogpro(task=-12.12537771913918), '7': SubLogpro(task=-4.53511688080016), '8': SubLogpro(task=-9.353306789860653), '9': SubLogpro(task=-15.059858856199199)})
    print('-----------------decomposed tasks------------------')
    #print(decomposed_tasks)
    # Convert the decomposed tasks into the internal processing format
    on_process_tasks = transform_task_data(decomposed_tasks.model_dump(), library)
    print(on_process_tasks)
    print('-------------------end of decomposed tasks-------------------')
    # --- c. Initialize the root node of the MCTS tree ---
    # Define the initial context
    pre_task = {'-1': {'task': '', 'code': '', 'score': 0, 'ori_id': '-1'}}
    curr_task = {'0': {'task': 'let\'s print hello to start.', 'code': 'import ee\nee.Initialize(project=\'ee-cyx669521\')\n', 'score': 1, 'ori_id': '0'}}
    on_process_tasks['0']['curr_task'] = curr_task['0']
    #print('-----------------on process tasks------------------')
    # Create the root node
    root = Node(logprob=log(1), overall_task=complete_task, current_task=curr_task, previous_task=pre_task, parent=None, label='default')
    root.visits += 1
    root.runs = 3
    root.update += 1
    root.finised_step_id = 0  # The root node represents the completion of the default first task
    root.value = 1.0  # Set the initial value of the root node to 1.0
    nodes, edges = {root.id: root}, {}
    graph_dict = {}
    i = 0
    best_fix_code = None
    best_fix_score = 0.0
    prompt_start = time.perf_counter()
    # [NEW] For tracking recorded final steps and building the final code structure
    final_code_structure = {}
    # Pre-record and build the initial step
    final_code_structure.update(root.current_task)
    record_final_step_decision(final_solution_log_filename, root)

    # The core of MCTS is to perform multiple "rollouts" to decide the best tree
    while i < max_rollouts and root.finised_step_id < len(on_process_tasks) - 1:
        i += 1
        current_task_info = on_process_tasks.get(str(root.finised_step_id), {}).get('curr_task', {})
        task_id = on_process_tasks.get(str(root.finised_step_id), {}).get('task_id', 'final')
        ori_task_id = current_task_info.get('ori_id', 'final')
        print('---------------------working on task_id:', task_id, ' ori_task_id:', ori_task_id, '---------------------')
        print(current_task_info)
        print(f"\n\n---- ROLLOUT {i}/{max_rollouts} | CURRENT TASK STEP: {root.finised_step_id}/{len(on_process_tasks)} ----")
        # --- 1. Selection ---
        # Starting from the root node, find a leaf node
        curr_node = root
        selection_path = [root.id]
        # As long as the current node has children, continue selecting downwards
        while curr_node._children:
            for child in curr_node._children:
                nodes[child.id] = child
                edges[(curr_node.id, child.id)] = True
            # Use the P-UCB strategy to select the optimal child node
            curr_node = p_ucb_select(curr_node, curr_node._children)
            selection_path.append(curr_node.id)
            # For any node selected as a leaf, increment its visit and run counts
            curr_node.visits += 1
            curr_node.runs += 1

        # Initialize the graph_dict entry
        rollout_key = f"rollout_{i}"

        # --- 2. Update ---
        # Check if the run count limit is exceeded (Hard Reset), reset subsequent tasks, and identity-map the current node
        #print('curr_node.runs, max_runs_per_node, curr_node.update', curr_node.runs, max_runs_per_node, curr_node.update)
        # This situation can only occur if the current node has failed multiple fix attempts without any progress,
        # i.e., when curr_node.runs > max_runs_per_node, curr_node.update == 0, and best_fix_score < 1.
        # This means the fix has failed, so we must re-plan from this step, mark this step as an identity map (update),
        # effectively skipping it, and continue with subsequent tasks.
        if curr_node.runs > max_runs_per_node and root.update < 3 and curr_node.update < 1 and best_fix_score < 1:
            print('--- Node Hard Reset Triggered ---')
            # Copy the original task state for logging
            original_task_for_log = copy.deepcopy(curr_node.current_task)

            #print('WATCH WATCH WATCH! I AM HERE!')
            # Select the best solution from all historical attempts
            best_fix_code, best_fix_score = get_best_program(curr_node.fix_attempts)
            if best_fix_code is None:
                best_fix_score = 1
                best_fix_code = 'print("update")\n'

            _pre_task_description = " ".join(task["task"] for task in curr_node.previous_task.values())
            _pre_task_code = "\n".join(task["code"] for task in curr_node.previous_task.values())
            _difficult_task = ('difficult task:' + curr_node.current_task[str(root.finised_step_id)]['task']
                               + 'its wrong code:' + curr_node.current_task[str(root.finised_step_id)]['code'])
            _decomposed_tasks = re_decompose_task(complete_task, curr_task=_difficult_task, pre_task=_pre_task_description)
            ##_decomposed_tasks = TaskOutput(tasks={'0': SubTask(task='print hello', code=''), '1': SubTask(task="initialize Earth Engine and define region of interest using asset path 'projects/ee-ecresener5/assets/jokkmokk'", code=''), '2': SubTask(task='define function to mask clouds and shadows using the SCL band', code=''), '3': SubTask(task='define function to calculate NDVI using B8 and B4 bands', code=''), '4': SubTask(task='create cloud-masked median composite for 2019 summer period (June 1 to August 31) over ROI, calculate NDVI, and rename bands with year', code=''), '5': SubTask(task='create cloud-masked median composite for 2020 summer period (June 1 to August 31) over ROI, calculate NDVI, and rename bands with year', code=''), '6': SubTask(task='create cloud-masked median composite for 2021 summer period (June 1 to August 31) over ROI, calculate NDVI, and rename bands with year', code=''), '7': SubTask(task='combine the three annual composite images into a single multi-band image', code=''), '8': SubTask(task='initialize interactive map and display the ROI and a true-color view of the 2021 composite', code=''), '9': SubTask(task="export the combined multi-band composite image to Google Drive as 'Annual_Summer_Composites_Jokkmokk' with 10-meter resolution and cloud-optimized GeoTIFF format", code='')}, transition_scores={'0': SubLogpro(task=-0.4159837565239286), '1': SubLogpro(task=-0.7280267480525993), '2': SubLogpro(task=-2.27854277176084), '3': SubLogpro(task=-1.9314369674923455), '4': SubLogpro(task=-9.400597509737622), '5': SubLogpro(task=-8.100786033280201), '6': SubLogpro(task=-12.12537771913918), '7': SubLogpro(task=-4.53511688080016), '8': SubLogpro(task=-9.353306789860653), '9': SubLogpro(task=-15.059858856199199)})
            #print(_decomposed_tasks)
            # Check if it's empty
            if not _decomposed_tasks.tasks:
                print('--- re-Decomposed Tasks failed by None output from LLM ---')
                continue
            _on_process_tasks = transform_task_data(_decomposed_tasks.model_dump(), library)
            #print('origin on_process_tasks', _on_process_tasks)
            # Assume dict1 and dict2 are the two dictionaries you want to concatenate
            merged_items = list(on_process_tasks.items())[:root.finised_step_id+1] + list(_on_process_tasks.items())[1:]
            #merged_items = list(on_process_tasks.items())[:root.finised_step_id] + list(_on_process_tasks.items())[root.finised_step_id:]
            # Renumber the keys
            print('----------------------updated tasks after re-decomposition----------------------')
            on_process_tasks = {str(i): v for i, (_, v) in enumerate(merged_items)}
            print(on_process_tasks)
            print('----------------------end of updated tasks after re-decomposition----------------------')
            #print('updated on_process_tasks', on_process_tasks)
            # Reset the current node
            _current_task_str = 'skip:' + curr_node.current_task[str(root.finised_step_id)]['task']
            _current_task_ori_id = curr_node.current_task[str(root.finised_step_id)]['ori_id'] + '_kill'
            curr_node.current_task = {str(root.finised_step_id): {'task': _current_task_str, 'code': best_fix_code, 'score': best_fix_score, 'ori_id': _current_task_ori_id}}
            curr_node.runs += 1
            curr_node.error = '' # To allow for the next expansion
            curr_node.update += 1 # No more decomposition
            root.update += 1
            # Update the Node label state
            curr_node.label = 'dead'
            # Identity transformation means the reward should be 1
            curr_node.backprop(best_fix_score)

            # Record this attempt in the node's history
            updated_task_for_log = copy.deepcopy(curr_node.current_task)
            updated_task_for_log['logs'] = {'re-decompose_tasks': on_process_tasks}
            details = {
                'ori_curr_task': original_task_for_log,
                'upd_curr_task': updated_task_for_log
            }
            record_rollout_state(rollout_key, root, selection_path, "decompose", action_details=details, save_filename=graph_log_filename)

        # -------3. Fix or Split-------
        # Check if this node has an error. If so, a task fix or decomposition is needed.
        #print(repr(curr_node.error))
        if curr_node.error:
            print('--- Node Error Detected ---')
            # Copy the original task state for logging
            original_task_for_log = copy.deepcopy(curr_node.current_task)
            # [Case A: Fix attempts not exhausted, attempt a fix in this rollout]
            if curr_node.runs <= max_runs_per_node:
                print(f"Node {curr_node.id} has an error. Attempting to fix (run {curr_node.runs}/{max_runs_per_node}).")
                # Task correction
                #print('curr_node.current_task', curr_node.current_task)
                updated_task = llm_task_update(curr_node.current_task, curr_node.error)
                print('-------------updated task from llm_task_update-------------')
                print(updated_task)
                print('-------------end of updated task from llm_task_update-------------')
                ##updated_task = TaskOutput(tasks={'0': SubTask(task='Authenticate Earth Engine access', code='import ee\ntry:\n    ee.Initialize()\nexcept Exception as e:\n    ee.Projection(\'EPSG:3857\')\n    ee.Initialize()'), '1': SubTask(task="Define the region of interest using the asset path 'projects/ee-ecresener5/assets/jokkmokk'", code="roi = ee.FeatureCollection('projects/ee-ecresener5/assets/jokkmokk')")}, transition_scores={'0': SubLogpro(task=-0.0027480495627969503), '1': SubLogpro(task=-0.11099041395209497)})
                #print('updated_task', updated_task)
                if not updated_task.tasks:
                    print("LLM failed to provide a fix. Skipping.")
                    continue

                fix_code = updated_task.tasks['0'].code
                __curr_task = curr_node.current_task
                __curr_task[str(root.finised_step_id)]['code'] = fix_code
                # Check if the updated existing task is complete
                #print(root.finised_step_id)
                print('curr_node.previous_task', curr_node.previous_task)
                print('__current_task', __curr_task)
                errors, pass_score = check_error_nodes(curr_node.previous_task, __curr_task, exec_path)
                print('-------------result after executing the fix code-------------')
                print('errors:', repr(errors))
                print('pass_score:', pass_score)
                print('-------------end of result after executing the fix code-------------')
                #errors = ' '
                #pass_score = 0.5
                #print(errors, pass_score) #'', 0.5
                print(f"  - Attempt execution scored: {pass_score:.4f}. Errors persist: {'Yes' if errors else 'No'}")

                if not errors:
                    # If the LLM split the task
                    if len(updated_task.tasks) > 1:
                        replace_task = updated_task.tasks['0'].task
                        replace_code = updated_task.tasks['0'].code
                        replace_score = updated_task.transition_scores['1'].task
                        replace_ori_id = curr_node.current_task[str(root.finised_step_id)]['ori_id'] + '_update'
                        insert_ori_id = curr_node.current_task[str(root.finised_step_id)]['ori_id'] + '_append'
                        curr_node.current_task = {
                            str(root.finised_step_id): {'task': replace_task, 'code': replace_code, 'score': replace_score, 'ori_id': replace_ori_id}}
                        insert_tasks = {}
                        __pre_task = curr_node.previous_task
                        for k, v in list(updated_task.tasks.items()):
                            __curr_task = {k: {'task': v.task, 'code': v.code, 'score': updated_task.transition_scores[k].task, 'ori_id': insert_ori_id}}
                            insert_tasks[k] = {
                                "task_id": '',
                                "pre_task": __pre_task,
                                "curr_task": __curr_task,
                                "library": '',
                            }
                            __pre_task = {**__pre_task, **__curr_task}
                        # len(updated_task.tasks) - 1 because one has replaced the current position
                        if len(on_process_tasks) - 1 > root.finised_step_id + len(updated_task.tasks) - 1:
                            on_process_tasks[str(root.finised_step_id + len(updated_task.tasks) - 1)]['pre_task'] = __pre_task
                        on_process_tasks.pop(str(root.finised_step_id), None)
                        on_process_tasks = insert_tasks_to_dict(insert_tasks, on_process_tasks, insert_after_id=root.finised_step_id - 1)
                else:
                    # If there are still errors, keep the current task unchanged, only update the code
                    curr_node.current_task[str(root.finised_step_id)]['code'] = fix_code
                    curr_node.current_task[str(root.finised_step_id)]['ori_id'] = curr_node.current_task[str(root.finised_step_id)]['ori_id'] + '_fix'

                #print('on_process_tasks:', on_process_tasks)
                # Record this attempt in the node's history
                curr_node.runs += 1
                curr_node.fix_attempts[fix_code] = pass_score
                # After correction, perform a simulation and backpropagate
                # Update Node label state
                curr_node.label = 'fix'
                curr_node.error = errors
                curr_node.backprop(pass_score)
                # Record this attempt in the node's history
                updated_task_for_log = copy.deepcopy(curr_node.current_task)
                updated_task_for_log['logs'] = {'updated_task': updated_task, 'error': errors}
                details = {
                    'ori_curr_task': original_task_for_log,
                    'upd_curr_task': updated_task_for_log
                }

                record_rollout_state(rollout_key, root, selection_path,
                                     "fix", action_details=details, save_filename=graph_log_filename)
                continue  # Proceed to the next rollout to re-evaluate this node
            else:
                # [Case B: Fix attempts are exhausted, make a final decision in this rollout]
                print(f"Node {curr_node.id} exhausted fix attempts. Resolving with best solution...")
                # Select the best solution from all historical attempts
                best_fix_code, best_fix_score = get_best_program(curr_node.fix_attempts)
                print('---------best fix code and score from the attempts-------')
                print(best_fix_code, best_fix_score)
                print('---------end of best fix code and score from the attempts-------')
                if best_fix_score is None:
                    best_fix_score = -1
                    best_fix_code = '# unable to fix the error.\nprint("skip")\n'
                curr_node.current_task[str(root.finised_step_id)]['code'] = best_fix_code
                curr_node.current_task[str(root.finised_step_id)]['score'] = best_fix_score
                curr_node.current_task[str(root.finised_step_id)]['ori_id'] = \
                curr_node.current_task[str(root.finised_step_id)]['ori_id'] + '_select'
                # Update Node label state
                curr_node.label = 'select'
                curr_node.backprop(best_fix_score)  # Backpropagate again, just in case
                # Record this attempt in the node's history
                updated_task_for_log = copy.deepcopy(curr_node.current_task)
                updated_task_for_log['logs'] = {'error': curr_node.error}
                details = {
                    'ori_curr_task': original_task_for_log,
                    'upd_curr_task': updated_task_for_log
                }
                record_rollout_state(rollout_key, root, selection_path,
                                     "fix", action_details=details, save_filename=graph_log_filename)


        # --- 4. Expansion ---
        # If the current node has no children, generate new children (new code candidates)
        #print(curr_node._children)
        if not curr_node._children:
            print('--- Node Expansion ---')
            # [CORE MODIFICATION]
            # When the algorithm decides to expand this node (curr_node), it means
            # the step it represents (root.finised_step_id) has a final solution. We record it here.
            print(curr_node.current_task)
            print('---------------------------------------------------------')
            print(f"--- COMMITTING to final solution for step {root.finised_step_id} ---")
            record_final_step_decision(
                final_solution_log_filename,
                curr_node,
            )
            # Update our final complete structure in real-time
            final_code_structure.update(curr_node.current_task)

            # Copy the original task state for logging
            original_task_for_log = copy.deepcopy(curr_node.current_task)
            root.finised_step_id += 1
            if root.finised_step_id < len(on_process_tasks) - 1:
                print(f"Expanding node {curr_node.id} for task step {root.finised_step_id}")
                next_task = on_process_tasks[str(root.finised_step_id)]
                next_logpro = next_task['curr_task']['score']
                print(next_task)
                pre_task = {**curr_node.previous_task, **curr_node.current_task}
                print(pre_task)
                code_candidates = generate_task_codes(pre_tasks=pre_task, curr_task=next_task['curr_task']['task'], k=top_k)
                ##code_candidates = TaskCodeWithCandidates(task="Initialize Earth Engine and define the region of interest using the asset path 'projects/ee-ecresener5/assets/jokkmokk'",
                ##                                         candidates=[CandidateCode(code="import ee\n\nee.Initialize()\nregion = ee.Image('projects/ee-ecresener5/assets/jokkmokk')\n", score=-4.144529103897071), CandidateCode(code="import ee\nee.Initialize()\nregion_of_interest = ee.FeatureCollection('projects/ee-ecresener5/assets/jokkmokk')\n", score=-1.8860520965763072), CandidateCode(code="import ee\nee.Initialize()\nroi = ee.FeatureCollection('projects/ee-ecresener5/assets/jokkmokk')\n", score=-0.9862874038541349)])
                if not code_candidates.candidates:
                    print('------------Expansion Failed by None output from LLM----------')
                    continue
                print('-------code candidates from the llm-------')
                print(code_candidates)
                print('-------end of code candidates from the llm-------')
                updated_task_for_log = {}
                for _idx, _candidate in enumerate(code_candidates.candidates):
                    code = _candidate.code
                    logpro = 0.5 * (_candidate.score + next_logpro)
                    _current_task = {str(root.finised_step_id): {'task': next_task['curr_task']['task'], 'code': code, 'score': logpro, 'ori_id': next_task['curr_task']['ori_id']}}
                    #print(_current_task)
                    #errors = check_child_nodes(pre_task, _current_task, exec_path)
                    errors, pass_score = check_error_nodes(pre_task, _current_task, exec_path)
                    print('-------get errors from the runtime check-----\n', errors, '\n-------score--------\n', pass_score)
                    child = Node(logprob=logpro, overall_task='', previous_task=pre_task,
                                            current_task=_current_task, parent=curr_node,
                                            finised_step_id=0, error=errors, label='expand')
                    child.fix_attempts[code] = pass_score
                    curr_node._children.append(child)
                    nodes[child.id] = child
                    edges[(curr_node.id, child.id)] = True
                    # Record the updated task state for logging
                    updated_task_for_log[str(_idx)] = {'task': _current_task, 'logs': {'error': errors, 'pass_score': pass_score}}

                # Record this attempt in the node's history
                details = {
                    'ori_curr_task': original_task_for_log,
                    'upd_curr_task': updated_task_for_log
                }
                record_rollout_state(rollout_key, root, selection_path,
                                     "expansion", action_details=details, save_filename=graph_log_filename)
            else:
                # The task is complete
                print("All task steps processed. Reached final state.")

        # --- 5. Simulation ---
        # Starting from the newly expanded node, calculate the reward through long exploration
        print('--- Node Simulation ---')
        sim_node = curr_node
        if curr_node._children:
            selected_child = p_ucb_select(curr_node, curr_node._children)
            sim_node = selected_child if selected_child else curr_node._children[0]
            selection_path.append(sim_node.id)

        # Copy the original task state for logging
        original_task_for_log = copy.deepcopy(sim_node.current_task)
        # Simulation is only meaningful if the node state is good (no errors)
        if not sim_node.error:
            print(f"Simulating from node {sim_node.id} (task step {sim_node.finised_step_id})")
            sim_pre_tasks = {**sim_node.previous_task, **sim_node.current_task}
            #print('sim_pre_tasks:', sim_pre_tasks)
            # Calculate the number of future steps to explore
            num_future_steps = max(0, min(3, len(on_process_tasks) - 1 - root.finised_step_id))
            # Generate the future tasks dictionary, only if there are tasks to generate
            long_explore_tasks = {}
            if num_future_steps > 0:
                long_explore_tasks = {
                    str(k): {'task': on_process_tasks[str(k)]['curr_task']['task'], 'code': ''}
                    for k in range(root.finised_step_id + 1, root.finised_step_id + 1 + num_future_steps)
                }
            completion_code = {}
            if long_explore_tasks is not None:
                full_code_output = generate_full_task_code(sim_pre_tasks, long_explore_tasks)
                if not full_code_output.tasks:
                    print('-----------Simulation Failed by LLM----------')
                    continue
                ##full_code_output = TaskOutput(tasks={'3': SubTask(task="Define a function to mask clouds and shadows using the 'SCL' band", code="def maskClouds(image):\n    scl = image.select('SCL')\n    cloud = scl.eq(1).or(scl.eq(8))\n    shadow = scl.eq(2).or(scl.eq(9))\n    mask = cloud.or(shadow).not()\n    return image.updateMask(mask)\n"), '4': SubTask(task="Define a function to calculate the Normalized Difference Vegetation Index (NDVI) using the 'B8' and 'B4' bands", code="def calculateNDVI(image):\n    return image.normalizedDifference(['B8', 'B4']).rename('ndvi')\n"), '5': SubTask(task="Filter Sentinel-2 images for 2019, apply cloud masking, create a median composite for the summer period (June 1 to August 31), calculate NDVI, and rename the NDVI band to 'ndvi_2019'", code="summer_images = ee.ImageCollection('COPERNICUS/S2_SR')\n    .filterDate('2019-06-01', '2019-08-31')\n    .filterBounds(roi)\nmasked_summer = summer_images.map(maskClouds)\nmedian_composite = masked_summer.median()\nndvi = calculateNDVI(median_composite).rename('ndvi_2019')\n")}, transition_scores={'3': SubLogpro(task=0.00041831703536345), '4': SubLogpro(task=0.08999356837137953), '5': SubLogpro(task=3.667576706362361e-06)})
                completion_code = taskoutput_to_dict(full_code_output)
            # The full simulation path includes the task of the sim_node itself
            full_completion = {**sim_node.current_task, **completion_code}
            print('---------------long exploration tasks---------------')
            print(long_explore_tasks)
            print(full_completion)
            print('---------------end of long exploration tasks---------------')
            reward = calculate_reward(sim_node.previous_task, full_completion, exec_path, len(on_process_tasks))
            print(f"Simulation from node {sim_node.id} received reward: {reward:.4f}")
            # Update Node label state
            sim_node.label = sim_node.label + '_simulate'
            sim_node.backprop(reward)
            # Record this attempt in the node's history
            updated_task_for_log = {'logs': {'long_explore_tasks': long_explore_tasks, 'full_completion': full_completion, 'reward': reward, 'error': sim_node.error}}
            details = {
                'ori_curr_task': original_task_for_log,
                'upd_curr_task': updated_task_for_log
            }
            record_rollout_state(rollout_key, root, selection_path,
                                 "simulate", action_details=details, save_filename=graph_log_filename)
        else:
            print(f"Skipping simulation from node {sim_node.id} due to existing errors.")
            # A failed simulation should also have a negative reward to avoid this path in the future
            reward = -1.0
            sim_node.label = sim_node.label + '_simulate'
            sim_node.backprop(reward)
            # Record this attempt in the node's history
            updated_task_for_log = {'logs': {'reward': reward, 'error': sim_node.error}}
            details = {
                'ori_curr_task': original_task_for_log,
                'upd_curr_task': updated_task_for_log
            }
            record_rollout_state(rollout_key, root, selection_path,
                                 "simulate", action_details=details, save_filename=graph_log_filename)
    # After the loop finishes, perform the final recording
    print("\n\n---- MCTS process finished ----")
    # [REPLACE] Instead of using record_task_result, directly output the final structure we built step-by-step
    print("\nFinal generated program structure:")
    final_full_code = ""
    # Ensure output is in step order
    sorted_keys = sorted(final_code_structure.keys(), key=int)
    for key in sorted_keys:
        task_info = final_code_structure[key]
        code_snippet = task_info.get('code', '')
        print(f"--- Step {key} (Task ID: {task_info.get('ori_id')}) ---")
        print(code_snippet)
        final_full_code += code_snippet + "\n"

    # Save the final complete script
    with open(f"{task_name}_final_script.py", "w", encoding="utf-8") as f:
        f.write(final_full_code)

    print(f"\nFinal solution steps logged to: {final_solution_log_filename}")
    print(f"MCTS trace logged to: {graph_log_filename}")
    print(f"Complete script saved to: {task_name}_final_script.py")
