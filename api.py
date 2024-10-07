"""
API functions for the GeoAgent
"""
import json
from typing import Any, Callable, Tuple, Union

from loguru import logger
import google.generativeai as genai
genai.configure(api_key='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', transport="rest")
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)
from datetime import datetime, timezone
from dataApi.gee_utils import BBox
from dataApi.geeData_tool import geeData_registery
from modelApi.samgeo_tools import samGeo_registry

logger.disable(__name__)


@retry(
    retry=retry_if_exception_type((ValueError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def chat_complete(
    task: str,
    sensor: str,
    task_type: str,
    query_interval: tuple[datetime.datetime, datetime.datetime],
    query_bbox: BBox,
    data_registry: Any = None,
    model_registry: Any = None
):
    data_functions = data_registry.to_list_infos(query_bbox=query_bbox, query_interval=query_interval, sensor=sensor) \
        if data_registry is not None else []

    if len(data_functions) == 0:
        raise UserWarning("No functions registered but expecting data functions")

    model_functions = model_registry.to_list_infos(sensor=sensor, task_type=task_type) \
        if model_registry is not None else []

    if len(model_functions) == 0:
        raise UserWarning("No functions registered but expecting model_functions")

    model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
    message_prompt = "I want you to become my Expert Python programmer. " \
                     f"Your goal is give corresponding code to the given task: {task}" \
                     + f'\nusing the potential data tools: {data_functions}.' \
                     + f'\nand the potential model tools: {model_functions}.' \
                     + "\nStart your code:"
    response = model.generate_content(message_prompt)
    code = response.candidates[0].content.parts[0].text
    logger.info(f"gemmini API returned: {code}")

    return code


if __name__ == '__main__':
    code = chat_complete(
        task='detect the wildfire area from image pairs',
        sensor='Sentinel-2',
        task_type='Change Detection',
        query_interval=(datetime(2020, 5, 17).replace(tzinfo=timezone.utc), datetime(2024, 5, 17).replace(tzinfo=timezone.utc)),
        query_bbox=BBox(2.3358203, 48.8421609, 2.3709914, 48.8624786),
        data_registry=geeData_registery,
        model_registry=samGeo_registry)
