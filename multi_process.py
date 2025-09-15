import multiprocessing
from multiprocessing import Manager
import time
import os
import psutil
from python_tool import PythonInterpreter
import sys
import io
import json

def make_serializable(obj):
    """
    递归处理不可序列化对象，将其替换为可序列化的值或移除。
    """
    if isinstance(obj, dict):
        # 如果是字典，递归处理其每个键值对
        return {k: make_serializable(v) for k, v in obj.items() if is_serializable(v)}
    elif isinstance(obj, list):
        # 如果是列表，递归处理其每个元素
        return [make_serializable(i) for i in obj if is_serializable(i)]
    elif isinstance(obj, tuple):
        # 如果是元组，转换为可序列化的列表
        return [make_serializable(i) for i in obj if is_serializable(i)]
    elif isinstance(obj, set):
        # 如果是集合，转换为可序列化的列表
        return [make_serializable(i) for i in obj if is_serializable(i)]
    elif is_serializable(obj):
        # 如果对象是可序列化的，直接返回
        return obj
    else:
        # 如果对象不可序列化，将其替换为字符串或移除
        return str(obj)


def is_serializable(obj):
    """
    检查对象是否可序列化为JSON格式。
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


#omly support unix system
'''
# 定义一个长时间运行的进程任务
def task(queue, code_snippet):

    python = PythonInterpreter(globals=globals(), locals=None)
    return_back_ = python.run(code_snippet)

    # Put the printed output and result into the queue
    queue.put(return_back_)


# 定义监控函数，检查进程的内存使用和执行时间
def monitor_process(code, max_time=10, max_memory_usage=90):
    if 'tensorflow' in code or 'keras' in code:
        return None

    # Create a Queue to capture the output and result
    queue = multiprocessing.Queue()

    # 启动进程
    proc = multiprocessing.Process(target=task, args=(queue, code))
    proc.start()

    start_time = time.time()

    while True:
        if not proc.is_alive():
            #print("Process has finished execution.")
            break

        # 检查执行时间
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > max_time:
            #print("Process exceeded maximum execution time. Killing process.")
            proc.terminate()  # 终止进程
            break

        # 检查内存使用
        try:
            process = psutil.Process(proc.pid)
            #memory_info = process.memory_info()
            #memory_usage = memory_info.rss / (1024 ** 2)  # 转换为 MB
            memory_percent = process.memory_percent()

            #print(f"Memory Usage: {memory_usage:.2f} MB, Memory Percent: {memory_percent:.2f}%, Elapsed Time: {elapsed_time:.2f} seconds")

            if memory_percent > max_memory_usage:
                #print("Process exceeded maximum memory usage. Killing process.")
                proc.terminate()  # 终止进程
                break
        except psutil.NoSuchProcess:
            #print("Process does not exist anymore.")
            break

        time.sleep(0.5)  # 每秒监控一次
    # 等待进程结束
    proc.join()
    print("Main program finished.")
    # Get the result and printed output from the queue
    result = queue.get() if not queue.empty() else None

    # Display the result
    print("Captured return value from target function:")
    print(result)
    return result
'''
def task(queue, code_snippet):
    # 注意：不能用 globals()，会触发 "cannot pickle 'module' object"
    python = PythonInterpreter(globals={}, locals={})
    result = python.run(code_snippet)
    queue.put(result)


def monitor_process(code, max_time=10, max_memory_usage=90):
    if 'tensorflow' in code or 'keras' in code:
        return None

    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=task, args=(queue, code))
    proc.start()

    start_time = time.time()
    while True:
        if not proc.is_alive():
            break

        if time.time() - start_time > max_time:
            proc.terminate()
            break

        try:
            process = psutil.Process(proc.pid)
            if process.memory_percent() > max_memory_usage:
                proc.terminate()
                break
        except psutil.NoSuchProcess:
            break

        time.sleep(0.5)

    proc.join()
    result = queue.get() if not queue.empty() else None
    return result




if __name__ == "__main__":
    # 设置监控的最大执行时间和内存使用百分比
    MAX_EXECUTION_TIME = 5  # 秒
    MAX_MEMORY_USAGE = 90  # 百分比

    code = 'import time\ntime.sleep(2)\nprintg'

    # 启动监控
    return_back = monitor_process(code, MAX_EXECUTION_TIME, MAX_MEMORY_USAGE)
    print(return_back)
