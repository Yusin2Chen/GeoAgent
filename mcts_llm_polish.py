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
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key='sk-or-v1-0fbb1ae0xxxxxxxxxxxxxxxxxxxxxxxxxxx')


def write_jsonl(filename: str, data: List[Union[Dict, list, str]], append: bool = False):
    """
    将数据写入 JSON Lines 文件。

    Args:
        filename: 文件路径
        data: 要写入的数据列表，每个元素通常是 dict
        append: 是否追加到文件末尾，如果 False 会覆盖文件
    """
    mode = "a" if append else "w"
    with open(filename, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

class Node:
    """
    蒙特卡洛搜索树 (MCTS) 的节点。

    每个节点代表一个特定的代码生成状态，存储了与该状态相关的所有信息，
    包括其父子关系、访问次数、价值、概率以及对应的任务代码。
    """
    # 使用 itertools.count() 为每个节点实例生成一个唯一的ID
    id_iter = itertools.count()

    def __init__(self, logprob, overall_task, current_task, previous_task, parent, finised_step_id=0, value=0, error='', label='default'):
        """
        初始化一个节点。

        Args:
            logprob (float): 该节点状态的对数概率。
            overall_task (str): 整个任务的描述。
            current_task (Dict): 当前节点代表的任务步骤及其代码。
            previous_task (Dict): 在此节点之前所有已完成的任务步骤和代码。
            parent (Node or None): 父节点。根节点的父节点为 None。
            finised_step_id (int): 已完成的任务序列中的步骤数量。
            value (float): 从该节点可获得的总奖励（Q值）。
            error (str): 如果该节点状态存在错误，记录错误信息。
        """
        self._children = []  # 子节点列表
        self._parent = parent  # 父节点引用
        self.visits = 1  # 节点被访问的次数 (N)
        self.runs = 0  # 节点被执行或尝试修复的次数
        self.finised_step_id = finised_step_id  # 已完成的任务步骤ID
        self.value = value  # 节点的价值或总奖励 (Q)
        self.prob = exp(logprob)  # 节点的先验概率 (P)，用于P-UCB计算
        self.overall_task = overall_task  # 整体任务描述
        self.current_task = current_task  # 当前步骤的任务和代码
        self.previous_task = previous_task  # 前置步骤的任务和代码
        self.id = next(self.id_iter)  # 节点的唯一ID
        self.p_ucb = 0  # 最新计算出的 P-UCB (Polynomial Upper Confidence Bound) 值
        self.error = error  # 记录子节点的错误信息

        # 【新增】存储对该节点的历次修复尝试，格式为 {code: score}
        self.fix_attempts = {}
        self.update = 0  # 记录节点被更新（如重新分解任务）的次数
        # 不知道有什么用
        self.label = label

    def backprop(self, value):
        self.visits += 1
        self.value += value  # 或者存 sum_rewards，然后用 value/visits 作为Q
        if self._parent is not None:
            self._parent.backprop(value)


class NodeEncoder(json.JSONEncoder):
    """
    自定义的 JSON 编码器，用于序列化 Node 对象。

    在序列化时，会移除对父子节点的直接引用，以避免循环依赖。
    """
    def default(self, obj):
        if isinstance(obj, Node):
            # 创建一个对象的浅拷贝，以避免修改原始对象
            cpy = copy.copy(obj)
            # 删除可能导致循环引用的属性
            del cpy._parent
            del cpy._children
            # 返回对象的 __dict__ 表示
            return vars(cpy)
        # 对于其他类型，使用基类的默认方法
        return json.JSONEncoder.default(self, obj)


def collect_graph_data(node: Node) -> Tuple[Dict[int, Node], List[Tuple[int, int]]]:
    """
    从根节点开始递归遍历，收集树中所有的节点和边，用于可视化。

    Args:
        node (Node): 开始遍历的根节点。

    Returns:
        Tuple[Dict[int, Node], List[Tuple[int, int]]]:
            - 一个字典，键是节点ID，值是节点对象。
            - 一个列表，包含所有表示父子关系的边 (父ID, 子ID)。
    """
    nodes = {node.id: node}
    edges = []
    # 遍历所有子节点
    for child in node._children:
        # 添加从当前节点到子节点的边
        edges.append((node.id, child.id))
        # 递归收集子树的节点和边
        child_nodes, child_edges = collect_graph_data(child)
        nodes.update(child_nodes)
        edges.extend(child_edges)
    return nodes, edges


def p_ucb_select(parent_node: Node, child_nodes: List[Node], c_base=10, c=4) -> Optional[Node]:
    """
    使用 P-UCB (Polynomial Upper Confidence Bound) 算法从子节点中选择一个。

    P-UCB 算法平衡了“利用”（exploitation，选择价值高的节点）和
    “探索”（exploration，选择访问次数少的节点）。

    公式: p_ucb = Q(s,a) + β * P(s,a) * sqrt(log(N(s))) / (1 + N(s,a))
    其中 β = log((N(s) + c_base + 1) / c_base) + c

    Args:
        parent_node (Node): 当前的父节点。
        child_nodes (List[Node]): 待选择的子节点列表。
        c_base (int): P-UCB 公式中的常数，影响探索权重。
        c (int): P-UCB 公式中的常数，影响探索权重。

    Returns:
        Node or None: P-UCB 值最高的子节点。如果列表为空，则返回 None。
    """
    s_visits = parent_node.visits  # 父节点的总访问次数 N(s)
    # 计算探索因子 beta
    beta = log((s_visits + c_base + 1) / c_base) + c
    #print(s_visits, beta)

    max_p_ucb = -inf
    max_node = None
    # 遍历所有子节点，计算其 P-UCB 值
    for node in child_nodes:
        # P-UCB 计算公式
        p_ucb = node.value + beta * node.prob * sqrt(log(s_visits)) / (1 + node.visits)
        #print(node.value, node.prob, s_visits, node.visits)

        # 打印调试信息
        print('-----------------------------------------selecting node---------------------------------')
        print(f"Node ID {node.id}: P-UCB = {p_ucb}")
        # 存储最新的P-UCB值，用于可视化
        node.p_ucb = p_ucb
        # 更新 P-UCB 值最高的节点
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
    根据任务步骤的执行成功率来计算奖励分数。

    此函数会先执行所有 pre_tasks 的代码来建立上下文，然后逐个执行
    completion 字典中每个步骤的代码。奖励基于 completion 中成功执行的步骤比例。
    如果所有步骤都成功且达到了任务总数，则返回满分 1.0。

    Args:
        pre_tasks (Dict): 一个字典，包含先前已完成的步骤。
                          格式: {'1': {'task': ..., 'code': ...}, ...}
        completion (Dict): 一个字典，包含需要被评估的新步骤。
                           格式: {'3': {'task': ..., 'code': ...}, ...}
        exec_path (str): 代码执行时的工作目录。
        full_steps (int): 完成整个任务所需的总步骤数。

    Returns:
        float: 奖励分数，范围在 0.0 到 1.0 之间。
    """
    # 如果 completion 为空，说明没有新的步骤需要评估，奖励为 0
    if not completion:
        return 0.0

    # 1. 准备执行环境和基础代码上下文
    # 设置工作目录，并处理可能的异常
    work_path_setup = f"import os\ntry:\n    os.chdir('{exec_path}')\nexcept Exception:\n    pass\n"

    # 拼接所有 pre_tasks 的代码作为初始上下文
    pre_tasks_code = ""
    # 确保 pre_tasks 按数字顺序拼接
    sorted_pre_keys = sorted(pre_tasks.keys(), key=int)
    for key in sorted_pre_keys:
        pre_tasks_code += pre_tasks[key].get('code', '') + '\n'

    # 2. 逐个步骤执行 completion 中的代码并计分
    successful_steps = 0
    cumulative_code = pre_tasks_code

    # 确保 completion 的步骤按数字顺序执行
    sorted_completion_keys = sorted(completion.keys(), key=int)

    for key in sorted_completion_keys:
        step_code = completion[key].get('code', '')
        # 将当前步骤的代码追加到累积代码中
        current_execution_code = cumulative_code + step_code + '\n'
        current_execution_code = autopep8.fix_code(current_execution_code)

        # 使用受监控的进程执行代码
        print('---------starting reward calculation by executing the full code for step', key, '---------')
        error_message = monitor_process(
            work_path_setup + current_execution_code,
            MAX_EXECUTION_TIME,
            MAX_MEMORY_USAGE
        )
        print('---------execution finished---------')

        # 判断执行是否成功
        if error_message is None:
            # 当前步骤成功，计分，并更新累积代码以备下一步使用
            successful_steps += 1
            cumulative_code = current_execution_code
        elif ('DeprecationWarning' in error_message) or ('ERROR' not in error_message) and ('Error' not in error_message) and ('Exception' not in error_message):
            # 某些非致命输出也可能被视为成功
            successful_steps += 1
            cumulative_code = current_execution_code
        else:
            # 一旦有步骤失败，立即停止，后续步骤不再执行
            print(f"--- Execution failed at step '{key}' and finished reward calculation ---")
            print(f"Code:\n{step_code.strip()}")
            print(f"Error: {error_message}")
            print('---------stop executing further steps---------')
            break

    # 3. 计算最终奖励
    # 奖励 = 成功执行的步骤数 / 总共尝试的步骤数
    reward = successful_steps / len(completion)

    # 4. 判断整个任务是否已彻底完成
    # 条件：(1) completion 中所有步骤都成功了 (2) 最后一个步骤的编号 >= 总步骤数
    all_completion_steps_succeeded = (successful_steps == len(completion))
    last_step_number = int(sorted_completion_keys[-1]) if sorted_completion_keys else -1

    if all_completion_steps_succeeded and last_step_number >= full_steps:
        print('************************** Full task completed successfully! *******************************')
        return 1.0  # 给予满分奖励

    return reward


def get_best_program(program_dict: Dict[str, float]) -> Tuple[Optional[str], float]:
    """
    从一个包含程序及其对应奖励的字典中，选出奖励最高的程序。

    Args:
        program_dict (Dict[str, float]): 一个字典，键是程序代码，值是其奖励分数。

    Returns:
        Tuple[Optional[str], float]:
            - 奖励最高的程序代码。如果字典为空，则为 None。
            - 对应的最高奖励分数。
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
    检查当前任务代码的错误，并计算当前代码自身的成功运行占比（score）。
    """

    found_errors = []

    # 1. 提取当前代码
    if not isinstance(curr_task, dict) or len(curr_task) != 1:
        return "输入错误: 'curr_task' 必须是包含一个条目的字典。", 0.0

    step_key = list(curr_task.keys())[0]
    new_code_snippet = curr_task[step_key].get('code', '').strip()

    if not new_code_snippet:
        return "", 0.0

    # 2. 语法检查
    try:
        tree = ast.parse(new_code_snippet)
    except SyntaxError as e:
        return f"语法错误 (SyntaxError): {e}", 0.0

    # 3. 准备前置代码和执行环境
    pre_code = ""
    sorted_pre_keys = sorted(pre_tasks.keys(), key=int)
    for key in sorted_pre_keys:
        pre_code += pre_tasks[key].get('code', '') + '\n'
    work_path_setup = f"import os\ntry:\n    os.chdir('{exec_path}')\nexcept Exception:\n    pass\n"

    # ----------- score 计算逻辑 -----------
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
            test_code_context += code_i  # 更新上下文
        else:
            error_message = return_back
            break  # 一旦报错就停止

    score = success_len / code_len if code_len > 0 else 0.0
    # ----------- score 计算结束 -----------

    if not error_message:
        return "", score

    # 4. 分析错误
    if 'NameError' in error_message:
        match = re.search(r"name '([^']*)' is not defined", error_message)
        if match:
            undefined_name = match.group(1)
            found_errors.append(f"未定义变量 '{undefined_name}'。")

    elif 'AttributeError' in error_message and jedi is not None:
        match = re.search(r"(?:module '|\w+' object|'(\w+)') has no attribute '(\w+)'", error_message)
        if match:
            obj_name, attr_name = match.groups()[-2:]
            found_errors.append(f"对象 '{obj_name}' 没有属性 '{attr_name}'。")
            try:
                script = jedi.Script(pre_code + new_code_snippet)
                lines = (pre_code + new_code_snippet).splitlines()
                last_line = len(lines)
                last_col = len(lines[-1]) if lines else 0
                completions = script.complete(line=last_line, column=last_col)
                obj_completions = [c.name for c in completions if c.name.startswith(attr_name[:2])]
                if obj_completions:
                    found_errors.append(f"是否想输入: {', '.join(obj_completions[:5])}?")
            except Exception as e:
                found_errors.append(f"Jedi 分析出错: {e}")


    elif ('DeprecationWarning' in error_message) or ('ERROR' in error_message) or ('Error' in error_message) or ('EEException' in error_message):
        found_errors.append(f"运行时错误: {error_message}")

    return "\n".join(found_errors), score



def check_child_nodes(
        pre_tasks: Dict[str, Dict[str, str]],
        curr_task: Dict[str, Dict[str, str]],
        exec_path: str
) -> str:
    """
    在给定先前任务上下文的情况下，检查当前任务代码的多种潜在错误。

    此函数会进行语法和运行时检查，并对特定错误（如 AttributeError）
    使用 jedi 提供额外信息。所有发现的错误将合并成一个字符串返回。

    Args:
        pre_tasks (Dict): 包含所有先前步骤的字典。
        curr_task (Dict): 包含当前待检查步骤的单元素字典。
        exec_path (str): 代码执行时的工作目录。

    Returns:
        str: 一个包含所有错误信息的字符串。如果代码有效，则返回空字符串。
    """
    found_errors = []

    # 1. 从输入字典中安全地提取代码
    if not isinstance(curr_task, dict) or len(curr_task) != 1:
        return "输入错误: 'curr_task' 必须是包含一个条目的字典。"

    step_key = list(curr_task.keys())[0]
    new_code_snippet = curr_task[step_key].get('code', '').strip()

    if not new_code_snippet:
        return ""  # 空代码视为有效

    # 2. 语法检查 (Syntax Check) - 这是第一道关卡
    try:
        ast.parse(new_code_snippet)
    except SyntaxError as e:
        # 语法错误是致命的，直接返回
        return f"语法错误 (SyntaxError): {e}"

    # 3. 准备完整的代码上下文用于运行时检查和 Jedi 分析
    pre_code = ""
    sorted_pre_keys = sorted(pre_tasks.keys(), key=int)
    for key in sorted_pre_keys:
        pre_code += pre_tasks[key].get('code', '') + '\n'

    full_code = pre_code + new_code_snippet
    full_code = autopep8.fix_code(full_code)
    work_path_setup = f"import os\ntry:\n    os.chdir('{exec_path}')\nexcept Exception:\n    pass\n"

    print('---------starting to execute the full code for runtime check---------')
    # 4. 运行时检查 (Runtime Check)
    error_message = monitor_process(
        work_path_setup + full_code,
        MAX_EXECUTION_TIME,
        MAX_MEMORY_USAGE
    )
    print('---------execution finished---------')

    #print(repr(error_message))

    if not error_message:
        return ""  # 没有错误，返回空字符串

    # 5. 深入分析捕获到的运行时错误
    # --- NameError 分析 ---
    if 'NameError' in error_message:
        match = re.search(r"name '([^']*)' is not defined", error_message)
        if match:
            undefined_name = match.group(1)
            found_errors.append(f"  - 详细分析: 检测到未定义的变量名 '{undefined_name}'。请检查变量是否已声明或拼写是否正确。")

    # --- AttributeError 分析 (使用 Jedi) ---
    elif 'AttributeError' in error_message and jedi is not None:
        match = re.search(r"(?:module '|\w+' object|'(\w+)') has no attribute '(\w+)'", error_message)
        if match:
            obj_name, attr_name = match.groups()[-2:]
            found_errors.append(f"  - 详细分析: 对象 '{obj_name}' 没有名为 '{attr_name}' 的属性。")
            try:
                script = jedi.Script(full_code)
                lines = full_code.splitlines()
                last_line = len(lines)
                last_col = len(lines[-1]) if lines else 0
                completions = script.complete(line=last_line, column=last_col)
                obj_completions = [c.name for c in completions if c.name.startswith(attr_name[:2])]
                if obj_completions:
                    suggestions = ", ".join(obj_completions[:5])
                    found_errors.append(f"  - Jedi 建议: 在 '{obj_name}' 上是否想输入: {suggestions}?")
                else:
                    found_errors.append(f"  - Jedi 建议: 未能在 '{obj_name}' 上找到与 '{attr_name}' 相关的属性。")
            except Exception as e:
                found_errors.append(f"  - Jedi 分析时出错: {e}")
    elif ('DeprecationWarning' in error_message) or ('ERROR' in error_message) or ('Error' in error_message) or ('EEException' in error_message):
        found_errors.append(f"运行时错误: {error_message}")

    # 6. 将所有收集到的错误信息合并成一个字符串
    return "\n".join(found_errors)


def transform_task_data(input_data, library_list):
    """
    将输入的任务字典转换为特定格式的字典，用于 MCTS 流程。

    Args:
      input_data (dict): 包含 "tasks" 键的原始输入字典。
      library_list (list): 要分配给每个任务的库列表。

    Returns:
      dict: 转换后包含任务信息的字典，键为任务ID。
    """
    tasks_list = {}
    previous_task = {}

    # 对任务ID进行排序，以确保它们按顺序处理
    sorted_task_ids = sorted(input_data['tasks'].keys(), key=int)

    for task_id_str in sorted_task_ids:
        task_id = str(task_id_str)  # 确保是字符串
        task_description = input_data['tasks'][task_id]['task']
        task_code = input_data['tasks'][task_id].get('code', '')
        task_score = input_data['transition_scores'][task_id].get('task', 0)


        # 构造当前步骤的 pre_task
        current_pre_task = copy.deepcopy(previous_task)

        transformed_task = {
            'task_id': task_id,
            'library': library_list,
            'curr_task': {'task': task_description, 'code': task_code, 'score': task_score, 'ori_id': task_id},
            'pre_task': current_pre_task,
        }
        tasks_list[task_id] = transformed_task

        # 更新 previous_task 以供下一个循环使用
        previous_task[task_id] = {'task': task_description, 'code': task_code, 'score': task_score, 'ori_id': task_id}

    return tasks_list

def taskoutput_to_dict(task_output: TaskOutput) -> dict:
    """Convert TaskOutput to dict with 'task', 'code', 'score' for each task."""
    result = {}
    sorted_task_ids = sorted(task_output.tasks.keys(), key=lambda x: int(x))  # 按数字顺序排序
    for task_id_str in sorted_task_ids:
        task_id = str(task_id_str)  # 确保是字符串
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
        tasks: Dict[str, Dict[str, Any]],  # 例如 {"1":{"curr_task": {...}, ...}, "2":{...}}
        on_process_tasks: Dict[str, Dict[str, Any]],  # 例如 {"0":{...}, "1":{...}}
        insert_after_id: int | str  # 在这个 id 之后插入
) -> Dict[str, Dict[str, Any]]:
    """
    将一组新任务插入到现有的任务处理队列中的指定位置之后。
    保留 curr_task 原有结构 (task/code/score)。

    Args:
        tasks (Dict): 要插入的新任务字典，每个值包含 curr_task。
        on_process_tasks (Dict): 当前的任务队列。
        insert_after_id (int | str): 在此任务ID之后插入新任务。

    Returns:
        Dict: 插入新任务后，重新编号的完整任务队列。
    """
    insert_after_id = str(insert_after_id)

    # 1. 把 on_process_tasks 转为按数字顺序的列表
    ordered_keys = sorted(on_process_tasks.keys(), key=int)
    task_list = [on_process_tasks[k] for k in ordered_keys]

    # 2. 定位插入位置
    if insert_after_id not in on_process_tasks:
        raise KeyError(f"insert_after_id '{insert_after_id}' 不在 on_process_tasks 里")
    pos = ordered_keys.index(insert_after_id)

    # 3. 生成要插入的新任务列表
    insert_list = [t for t in tasks.values()]
    # 4. 在指定位置后拼接列表
    new_list = task_list[:pos + 1] + insert_list + task_list[pos + 1:]

    # 5. 重新编号并还原为字典
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
    记录一个已确定的任务步骤的最终决策。
    """
    # 从决策节点中提取当前步骤的信息
    item = dict(
        curr_task=decision_node.current_task,
        # 从节点中获取一些元数据
        node_id=decision_node.id,
        node_value=decision_node.value,
        node_visits=decision_node.visits
    )
    write_jsonl(filename, [item], append=True)


# 新版本 - 请使用这个
def record_rollout_state(
        rollout_key: str, root: Node, selection_path: List[int],
        action: str,
        action_details: Optional[Dict] = None,
        save_filename: str = None
):
    """
    捕获当前MCTS树的快照，并将其作为一条记录追加到日志文件中。
    """
    if not save_filename:
        print("警告: 未向 record_rollout_state 提供 save_filename。状态未保存。")
        return

    # 1. 收集图的当前快照
    all_nodes, all_edges = collect_graph_data(root)
    # 【关键改动】: 不再手动调用 make_serializable。直接使用原始的 all_nodes 字典。
    # NodeEncoder 会在下一步的 json.dumps 中处理它们。

    # 2. 构造要写入文件的数据
    rollout_data = {
        "action": action,
        "selected_nodes_path": selection_path,
        # 这里的 make_serializable 可能是必要的，如果 action_details 包含其他复杂对象
        "action_details": make_serializable(action_details) if action_details else {},
        "tree_snapshot": {"nodes": all_nodes, "edges": all_edges} # <--- 直接传递 all_nodes
    }

    # 3. 将这条记录追加到 .jsonl 文件
    save_rollout_snapshot_to_jsonl(save_filename, rollout_key, rollout_data)


# 新版本 - 请使用这个
def save_rollout_snapshot_to_jsonl(filename: str, rollout_key: str, data: Dict):
    """将单次rollout的数据作为一行追加到JSON Lines文件，并始终使用NodeEncoder。"""
    line_data = {"rollout_key": rollout_key, **data}
    try:
        with open(filename, "a", encoding="utf-8") as f:
            # 直接、唯一地使用 NodeEncoder 来处理序列化
            f.write(json.dumps(line_data, cls=NodeEncoder, ensure_ascii=False, indent=None) + "\n")
    except TypeError as e:
        # 如果还是出错，现在我们会看到明确的错误信息，而不是被静默处理
        print(f"!!! 序列化日志时发生严重错误: {e}")
        # 可以考虑在这里写入一个错误标记，而不是格式错误的数据
        error_line = {"rollout_key": rollout_key, "error": f"Serialization failed: {e}"}
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_line) + "\n")


if __name__ == "__main__":
    task_path = 'D:\\pythonCode\\pythonCode\\MCTS_Agent\\workspace\\GEE_0004.json'
    output_doc = 'D:\\pythonCode\\pythonCode\\MCTS_Agent\\workspace\\GEE_0004_multi.json'
    exec_path = './workspace'
    library = ['numpy', 'geemap', 'ee']
    task_name = os.path.basename(task_path).split('.')[0]
    # 【新增】定义清晰分离的日志文件
    final_solution_log_filename = f"{task_name}_solution_steps.jsonl"
    graph_log_filename = f"graph_{task_name}_rollout_log.jsonl"  # 保持不变
    # 【新增】在开始前清空旧的最终结果日志
    if os.path.exists(final_solution_log_filename):
        os.remove(final_solution_log_filename)
    if os.path.exists(graph_log_filename):
        os.remove(graph_log_filename)

    # MCTS 超参数
    MAX_EXECUTION_TIME = 10  # 秒
    MAX_MEMORY_USAGE = 512  # MB
    max_rollouts = 20  # 最大推演次数
    max_runs_per_node = 2  # 每个节点的最大尝试/修复次数
    top_k = 3  # 每次扩展时生成的代码候选数量

    # --- b. 加载并分解初始任务 ---
    tasks = json.load(open(task_path))
    task_list = [v['task'] for k, v in tasks.items()]
    complete_task = '\n'.join(task_list)
    # 调用 LLM 进行任务分解
    decomposed_tasks = decompose_task(complete_task, pre_task='let\'s print hello to start.')
    if not decomposed_tasks.tasks:
        print('----------Task Decomposition Failed----------')
    ##decomposed_tasks = TaskOutput(tasks={'0': SubTask(task='print hello', code=''), '1': SubTask(task="initialize Earth Engine and define region of interest using asset path 'projects/ee-ecresener5/assets/jokkmokk'", code=''), '2': SubTask(task='define function to mask clouds and shadows using the SCL band', code=''), '3': SubTask(task='define function to calculate NDVI using B8 and B4 bands', code=''), '4': SubTask(task='create cloud-masked median composite for 2019 summer period (June 1 to August 31) over ROI, calculate NDVI, and rename bands with year', code=''), '5': SubTask(task='create cloud-masked median composite for 2020 summer period (June 1 to August 31) over ROI, calculate NDVI, and rename bands with year', code=''), '6': SubTask(task='create cloud-masked median composite for 2021 summer period (June 1 to August 31) over ROI, calculate NDVI, and rename bands with year', code=''), '7': SubTask(task='combine the three annual composite images into a single multi-band image', code=''), '8': SubTask(task='initialize interactive map and display the ROI and a true-color view of the 2021 composite', code=''), '9': SubTask(task="export the combined multi-band composite image to Google Drive as 'Annual_Summer_Composites_Jokkmokk' with 10-meter resolution and cloud-optimized GeoTIFF format", code='')}, transition_scores={'0': SubLogpro(task=-0.4159837565239286), '1': SubLogpro(task=-0.7280267480525993), '2': SubLogpro(task=-2.27854277176084), '3': SubLogpro(task=-1.9314369674923455), '4': SubLogpro(task=-9.400597509737622), '5': SubLogpro(task=-8.100786033280201), '6': SubLogpro(task=-12.12537771913918), '7': SubLogpro(task=-4.53511688080016), '8': SubLogpro(task=-9.353306789860653), '9': SubLogpro(task=-15.059858856199199)})
    print('-----------------decomposed tasks------------------')
    #print(decomposed_tasks)
    # 将分解后的任务转换为内部处理格式
    on_process_tasks = transform_task_data(decomposed_tasks.model_dump(), library)
    print(on_process_tasks)
    print('-------------------end of decomposed tasks-------------------')
    # --- c. 初始化 MCTS 树的根节点 ---
    # 定义初始上下文
    pre_task = {'-1': {'task': '', 'code': '', 'score': 0, 'ori_id': '-1'}}
    curr_task = {'0': {'task': 'let\'s print hello to start.', 'code': 'import ee\nee.Initialize(project=\'ee-cyx669521\')\n', 'score': 1, 'ori_id': '0'}}
    on_process_tasks['0']['curr_task'] = curr_task['0']
    #print('-----------------on process tasks------------------')
    # 创建根节点
    root = Node(logprob=log(1), overall_task=complete_task, current_task=curr_task, previous_task=pre_task, parent=None, label='default')
    root.visits += 1
    root.runs = 3
    root.update += 1
    root.finised_step_id = 0  # 根节点表示完成了默认开始的第一个任务
    root.value = 1.0  # 根节点初始价值设为 1.0
    nodes, edges = {root.id: root}, {}
    graph_dict = {}
    i = 0
    best_fix_code = None
    best_fix_score = 0.0
    prompt_start = time.perf_counter()
    # 【新增】用于跟踪已记录的最终步骤和构建最终代码结构
    final_code_structure = {}
    # 预先记录并构建初始步骤
    final_code_structure.update(root.current_task)
    record_final_step_decision(final_solution_log_filename, root)

    # MCTS的核心是进行多次“摇摆”或“推演”（rollouts）来决定最好的树
    while i < max_rollouts and root.finised_step_id < len(on_process_tasks) - 1:
        i += 1
        current_task_info = on_process_tasks.get(str(root.finised_step_id), {}).get('curr_task', {})
        task_id = on_process_tasks.get(str(root.finised_step_id), {}).get('task_id', 'final')
        ori_task_id = current_task_info.get('ori_id', 'final')
        print('---------------------working on task_id:', task_id, ' ori_task_id:', ori_task_id, '---------------------')
        print(current_task_info)
        print(f"\n\n---- ROLLOUT {i}/{max_rollouts} | CURRENT TASK STEP: {root.finised_step_id}/{len(on_process_tasks)} ----")
        # --- 1. 选择 (Selection) ---
        # 从根节点开始，找到一个叶子节点
        curr_node = root
        selection_path = [root.id]
        # 只要当前节点有子节点，就继续向下选择
        while curr_node._children:
            for child in curr_node._children:
                nodes[child.id] = child
                edges[(curr_node.id, child.id)] = True
            # 使用P-UCB策略选择最优的子节点
            curr_node = p_ucb_select(curr_node, curr_node._children)
            selection_path.append(curr_node.id)
            # 任何被选为叶节点的节点，访问和运行次数都增加
            curr_node.visits += 1
            curr_node.runs += 1

        # 初始化 graph_dict 条目
        rollout_key = f"rollout_{i}"

        # --- 2. 更新任务 (Update) ---
        # 检查运行次数是否超限 (Hard Reset), 重置后续任务，等变映射当前节点
        #print('curr_node.runs, max_runs_per_node, curr_node.update', curr_node.runs, max_runs_per_node, curr_node.update)
        # 这种情况只可能是因为当前节点多次尝试修复失败，且没有任何进展， 也就是说在curr_node.runs > max_runs_per_node的情况下， curr_node.update == 0而且best_fix_score < 1.
        # 也就是说修复失败了，那么我们只能从这一步重新规划，并且将这一步变为恒等映射并标记为update，也就是跳过这一步，继续后续的任务
        if curr_node.runs > max_runs_per_node and root.update < 3 and curr_node.update < 1 and best_fix_score < 1:
            print('--- Node Hard Reset Triggered ---')
            # 拷贝原始任务状态用于日志
            original_task_for_log = copy.deepcopy(curr_node.current_task)

            #print('WATCH WATCH WATCH! I AM HERE!')
            # 从所有历史尝试中选出最优解
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
            # 判断是否是空
            if not _decomposed_tasks.tasks:
                print('--- re-Decomposed Tasks failed by None output from LLM ---')
                continue
            _on_process_tasks = transform_task_data(_decomposed_tasks.model_dump(), library)
            #print('origin on_process_tasks', _on_process_tasks)
            # 假设 dict1 和 dict2 是你要拼接的两个字典
            merged_items = list(on_process_tasks.items())[:root.finised_step_id+1] + list(_on_process_tasks.items())[1:]
            #merged_items = list(on_process_tasks.items())[:root.finised_step_id] + list(_on_process_tasks.items())[root.finised_step_id:]
            # 重新编号 key
            print('----------------------updated tasks after re-decomposition----------------------')
            on_process_tasks = {str(i): v for i, (_, v) in enumerate(merged_items)}
            print(on_process_tasks)
            print('----------------------end of updated tasks after re-decomposition----------------------')
            #print('updated on_process_tasks', on_process_tasks)
            # 重置当前节点
            _current_task_str = 'skip:' + curr_node.current_task[str(root.finised_step_id)]['task']
            _current_task_ori_id = curr_node.current_task[str(root.finised_step_id)]['ori_id'] + '_kill'
            curr_node.current_task = {str(root.finised_step_id): {'task': _current_task_str, 'code': best_fix_code, 'score': best_fix_score, 'ori_id': _current_task_ori_id}}
            curr_node.runs += 1
            curr_node.error = '' # 为了扩展下一步
            curr_node.update += 1 # 不再进行分解
            root.update += 1
            # 更新Node label状态
            curr_node.label = 'dead'
            # 恒等变幻即reward应当是1
            curr_node.backprop(best_fix_score)

            # 记录本次尝试到节点的历史记录中
            updated_task_for_log = copy.deepcopy(curr_node.current_task)
            updated_task_for_log['logs'] = {'re-decompose_tasks': on_process_tasks}
            details = {
                'ori_curr_task': original_task_for_log,
                'upd_curr_task': updated_task_for_log
            }
            record_rollout_state(rollout_key, root, selection_path, "decompose", action_details=details, save_filename=graph_log_filename)

        # -------3. 修复或分解 (fix or split)-------
        # 判断这个节点是否有error, 如果有error则需要进行任务修正或者任务分解
        #print(repr(curr_node.error))
        if curr_node.error:
            print('--- Node Error Detected ---')
            # 拷贝原始任务状态用于日志
            original_task_for_log = copy.deepcopy(curr_node.current_task)
            # 【情况A：修复次数未满，本次 rollout 进行一次修复尝试】
            if curr_node.runs <= max_runs_per_node:
                print(f"Node {curr_node.id} has an error. Attempting to fix (run {curr_node.runs}/{max_runs_per_node}).")
                # 任务修正
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
                # 判断更新的现有任务是否完成
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
                    # 如果 LLM 将任务拆分了
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
                        #len(updated_task.tasks) - 1 是因为有一个替换了当前的位置
                        if len(on_process_tasks) - 1 > root.finised_step_id + len(updated_task.tasks) - 1:
                            on_process_tasks[str(root.finised_step_id + len(updated_task.tasks) - 1)]['pre_task'] = __pre_task
                        on_process_tasks.pop(str(root.finised_step_id), None)
                        on_process_tasks = insert_tasks_to_dict(insert_tasks, on_process_tasks, insert_after_id=root.finised_step_id - 1)
                else:
                    # 如果仍有错误，保留当前任务不变，只更新代码
                    curr_node.current_task[str(root.finised_step_id)]['code'] = fix_code
                    curr_node.current_task[str(root.finised_step_id)]['ori_id'] = curr_node.current_task[str(root.finised_step_id)]['ori_id'] + '_fix'

                #print('on_process_tasks:', on_process_tasks)
                # 记录本次尝试到节点的历史记录中
                curr_node.runs += 1
                curr_node.fix_attempts[fix_code] = pass_score
                # 修正后，进行一次模拟并反向传播
                # 更新Node label 状态
                curr_node.label = 'fix'
                curr_node.error = errors
                curr_node.backprop(pass_score)
                # 记录本次尝试到节点的历史记录中
                updated_task_for_log = copy.deepcopy(curr_node.current_task)
                updated_task_for_log['logs'] = {'updated_task': updated_task, 'error': errors}
                details = {
                    'ori_curr_task': original_task_for_log,
                    'upd_curr_task': updated_task_for_log
                }

                record_rollout_state(rollout_key, root, selection_path,
                                     "fix", action_details=details, save_filename=graph_log_filename)
                continue  # 直接进入下一次 rollout，重新评估此节点
            else:
                # 【情况B：修复次数已满，本次 rollout 进行最终决策】
                print(f"Node {curr_node.id} exhausted fix attempts. Resolving with best solution...")
                # 从所有历史尝试中选出最优解
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
                # 更新Node label 状态
                curr_node.label = 'select'
                curr_node.backprop(best_fix_score)  # 再次反向传播，以防万一
                # 记录本次尝试到节点的历史记录中
                updated_task_for_log = copy.deepcopy(curr_node.current_task)
                updated_task_for_log['logs'] = {'error': curr_node.error}
                details = {
                    'ori_curr_task': original_task_for_log,
                    'upd_curr_task': updated_task_for_log
                }
                record_rollout_state(rollout_key, root, selection_path,
                                     "fix", action_details=details, save_filename=graph_log_filename)


        # --- 4. 扩展 (Expansion) ---
        # 如果当前节点没有子节点，就生成新的子节点（新的代码候选）
        #print(curr_node._children)
        if not curr_node._children:
            print('--- Node Expansion ---')
            # 【核心修改点】
            # 在算法决定要扩展这个节点(curr_node)时，意味着它所代表的步骤
            # (root.finised_step_id) 已经有了最终解。我们在这里记录它。
            print(curr_node.current_task)
            print('---------------------------------------------------------')
            print(f"--- COMMITTING to final solution for step {root.finised_step_id} ---")
            record_final_step_decision(
                final_solution_log_filename,
                curr_node,
            )
            # 实时更新我们的最终完整结构
            final_code_structure.update(curr_node.current_task)

            # 拷贝原始任务状态用于日志
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
                    # 记录更新后的任务状态用于日志
                    updated_task_for_log[str(_idx)] = {'task': _current_task, 'logs': {'error': errors, 'pass_score': pass_score}}

                # 记录本次尝试到节点的历史记录中
                details = {
                    'ori_curr_task': original_task_for_log,
                    'upd_curr_task': updated_task_for_log
                }
                record_rollout_state(rollout_key, root, selection_path,
                                     "expansion", action_details=details, save_filename=graph_log_filename)
            else:
                # 任务已完成
                print("All task steps processed. Reached final state.")

        # --- 5. 模拟 (Simulation) ---
        # 从新扩展的节点开始，通过long exploration计算奖励
        print('--- Node Simulation ---')
        sim_node = curr_node
        if curr_node._children:
            selected_child = p_ucb_select(curr_node, curr_node._children)
            sim_node = selected_child if selected_child else curr_node._children[0]
            selection_path.append(sim_node.id)

        # 拷贝原始任务状态用于日志
        original_task_for_log = copy.deepcopy(sim_node.current_task)
        # 只有在节点状态良好（无错误）时，模拟才有意义
        if not sim_node.error:
            print(f"Simulating from node {sim_node.id} (task step {sim_node.finised_step_id})")
            sim_pre_tasks = {**sim_node.previous_task, **sim_node.current_task}
            #print('sim_pre_tasks:', sim_pre_tasks)
            # 计算未来要探索的步骤数量
            num_future_steps = max(0, min(3, len(on_process_tasks) - 1 - root.finised_step_id))
            # 生成未来任务字典，仅在有任务时才生成
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
            # 完整的模拟路径包括 sim_node 自身的任务
            full_completion = {**sim_node.current_task, **completion_code}
            print('---------------long exploration tasks---------------')
            print(long_explore_tasks)
            print(full_completion)
            print('---------------end of long exploration tasks---------------')
            reward = calculate_reward(sim_node.previous_task, full_completion, exec_path, len(on_process_tasks))
            print(f"Simulation from node {sim_node.id} received reward: {reward:.4f}")
            # 更新Node label 状态
            sim_node.label = sim_node.label + '_simulate'
            sim_node.backprop(reward)
            # 记录本次尝试到节点的历史记录中
            updated_task_for_log = {'logs': {'long_explore_tasks': long_explore_tasks, 'full_completion': full_completion, 'reward': reward, 'error': sim_node.error}}
            details = {
                'ori_curr_task': original_task_for_log,
                'upd_curr_task': updated_task_for_log
            }
            record_rollout_state(rollout_key, root, selection_path,
                                 "simulate", action_details=details, save_filename=graph_log_filename)
        else:
            print(f"Skipping simulation from node {sim_node.id} due to existing errors.")
            # 模拟失败也要算一个负数的reward吧，让其下次跳坑啊
            reward = -1.0
            sim_node.label = sim_node.label + '_simulate'
            sim_node.backprop(reward)
            # 记录本次尝试到节点的历史记录中
            updated_task_for_log = {'logs': {'reward': reward, 'error': sim_node.error}}
            details = {
                'ori_curr_task': original_task_for_log,
                'upd_curr_task': updated_task_for_log
            }
            record_rollout_state(rollout_key, root, selection_path,
                                 "simulate", action_details=details, save_filename=graph_log_filename)
    # 循环结束后，进行最终的记录
    print("\n\n---- MCTS process finished ----")
    # 【替换】不再使用 record_task_result，而是直接输出我们逐步构建的最终结构
    print("\nFinal generated program structure:")
    final_full_code = ""
    # 确保按步骤顺序输出
    sorted_keys = sorted(final_code_structure.keys(), key=int)
    for key in sorted_keys:
        task_info = final_code_structure[key]
        code_snippet = task_info.get('code', '')
        print(f"--- Step {key} (Task ID: {task_info.get('ori_id')}) ---")
        print(code_snippet)
        final_full_code += code_snippet + "\n"

    # 保存最终的完整脚本
    with open(f"{task_name}_final_script.py", "w", encoding="utf-8") as f:
        f.write(final_full_code)

    print(f"\nFinal solution steps logged to: {final_solution_log_filename}")
    print(f"MCTS trace logged to: {graph_log_filename}")
    print(f"Complete script saved to: {task_name}_final_script.py")
