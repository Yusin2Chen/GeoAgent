# Standard library imports
import base64
import contextlib
from concurrent import futures
from contextlib import redirect_stdout
import dataclasses
import datetime
import enum
import io
from io import BytesIO
import json
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
import traceback
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence
# Third-party imports
import dateutil
#import ee
#import geemap
import IPython
from IPython.display import HTML, Javascript, display, clear_output
from ipyleaflet import LayerException
import ipywidgets as widgets
import iso8601
from jinja2 import Template
import numpy as np
import pandas as pd
from PIL import Image
import requests
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tqdm


def matches_interval(
    collection_interval: tuple[datetime.datetime, datetime.datetime],
    query_interval: tuple[datetime.datetime, datetime.datetime],
):
  """Checks if the collection's datetime interval matches the query datetime interval.

  Args:
    collection_interval: Temporal interval of the collection.
    query_interval: a tuple with the query interval start and end

  Returns:
    True if the datetime interval matches
  """
  start_query, end_query = query_interval
  start_collection, end_collection = collection_interval
  if end_collection is None:
    # End date should always be set in STAC JSON files, but just in case...
    end_collection = datetime.datetime.now(tz=datetime.UTC)
  return end_query > start_collection and start_query <= end_collection


def matches_datetime(
    collection_interval: tuple[datetime.datetime, Optional[datetime.datetime]],
    query_datetime: datetime.datetime,
):
  """Checks if the collection's datetime interval matches the query datetime.

  Args:
    collection_interval: Temporal interval of the collection.
    query_datetime: a datetime coming from a query

  Returns:
    True if the datetime interval matches
  """
  if collection_interval[1] is None:
    # End date should always be set in STAC JSON files, but just in case...
    end_date = datetime.datetime.now(tz=datetime.UTC)
  else:
    end_date = collection_interval[1]
  return collection_interval[0] <= query_datetime <= end_date


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(1),
    retry=tenacity.retry_if_exception_type(LayerException),
    # before_sleep=lambda retry_state: print(f"LayerException occurred. Retrying in 1 seconds... (Attempt {retry_state.attempt_number}/3)")
)
def run_ee_code(code: str, ee, geemap_instance: Any):
    try:
        # geemap appears to have some stray print statements.
      _ = io.StringIO()
      with redirect_stdout(_):
        # Note that sometimes the geemap code uses both 'Map' and 'm' to refer to a map instance.
        exec(code, {'ee': ee, 'Map': geemap_instance, 'm': geemap_instance})
    except Exception:
        # Re-raise the exception with the original traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        raise exc_value.with_traceback(exc_traceback)


@dataclasses.dataclass
class BBox:
  """Class representing a lat/lon bounding box."""
  west: float
  south: float
  east: float
  north: float

  def is_global(self) -> bool:
    return (
        self.west == -180 and self.south == -90 and
        self.east == 180 and self.north == 90)

  @classmethod
  def from_list(cls, bbox_list: list[float]):
    """Constructs a BBox from a list of four numbers [west,south,east,north]."""
    if bbox_list[0] > bbox_list[2]:
      raise ValueError(
          'The smaller (west) coordinate must be listed first in a bounding box'
          f' corner list. Found {bbox_list}'
      )
    if bbox_list[1] > bbox_list[3]:
      raise ValueError(
          'The smaller (south) coordinate must be listed first in a bounding'
          f' box corner list. Found {bbox_list}'
      )
    return cls(bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3])

  def to_list(self) -> list[float]:
    return [self.west, self.south, self.east, self.north]

  def intersects(self, query_bbox) -> bool:
    """Checks if this bbox intersects with the query bbox.

    Doesn't handle bboxes extending past the antimeridaian.

    Args:
      query_bbox: Bounding box from the query.

    Returns:
      True if the two bounding boxes intersect
    """
    return (
        query_bbox.west < self.east
        and query_bbox.east > self.west
        and query_bbox.south < self.north
        and query_bbox.north > self.south
    )


# @title class Collection()
class Collection:
  """A simple wrapper for a STAC Collection.."""
  stac_json: dict[str, Any]

  def __init__(self, stac_json: dict[str, Any]):
    self.stac_json = stac_json
    if stac_json.get('gee:status') == 'deprecated':
      # Set the STAC 'deprecated' field that we don't set in the jsonnet files
      stac_json['deprecated'] = True

  def __getitem__(self, item: str) -> Any:
    return self.stac_json[item]

  def get(self, item: str, default: Optional[Any] = None) -> Optional[Any]:
    """Matches dict's get by returning None if there is no item."""
    return self.stac_json.get(item, default)

  def public_id(self) -> str:
    return self['id']

  def hyphen_id(self) -> str:
    return self['id'].replace('/', '_')

  def get_dataset_type(self) -> str:
    """Could be Image, ImageCollection, FeatureCollection, Feature."""
    return self['gee:type']

  def is_deprecated(self) -> bool:
    """Returns True for collections that are deprecated or have a successor."""
    if self.get('deprecated', False):
      logging.info('Skipping deprecated collection: %s', self.public_id())
      return True

  def datetime_interval(
      self,
  ) -> Iterable[tuple[datetime.datetime, Optional[datetime.datetime]]]:
    """Returns datetime objects representing temporal extents."""
    for stac_interval in self.stac_json['extent']['temporal']['interval']:
      if not stac_interval[0]:
        raise ValueError(
            'Expected a non-empty temporal interval start for '
            + self.public_id()
        )
      start_date = iso8601.parse_date(stac_interval[0])
      if stac_interval[1] is not None:
        end_date = iso8601.parse_date(stac_interval[1])
      else:
        end_date = None
      yield (start_date, end_date)

  def start(self) -> datetime.datetime:
    return list(self.datetime_interval())[0][0]

  def start_str(self) -> datetime.datetime:
    if not self.start():
      return ''
    return self.start().strftime("%Y-%m-%d")

  def end(self) -> Optional[datetime.datetime]:
    return list(self.datetime_interval())[0][1]

  def end_str(self) -> Optional[datetime.datetime]:
    if not self.end():
      return ''
    return self.end().strftime("%Y-%m-%d")

  def bbox_list(self) -> Sequence[BBox]:
    if 'extent' not in self.stac_json:
      # Assume global if nothing listed.
      return (BBox(-180, -90, 180, 90),)
    return tuple([
        BBox.from_list(x)
        for x in self.stac_json['extent']['spatial']['bbox']
    ])

  def bands(self) -> List[Dict]:
    summaries = self.stac_json.get('summaries')
    if not summaries:
      return []
    return summaries.get('eo:bands', [])

  def spatial_resolution_m(self) -> float:
    summaries = self.stac_json.get('summaries')
    if not summaries:
      return -1
    if summaries.get('gsd'):
      return summaries.get('gsd')[0]

    # Fallback for cases where the stac does not follow convention.
    gsd_lst = re.findall(r'"gsd": (\d+)', json.dumps(self.stac_json))

    if len(gsd_lst) > 0:
      return float(gsd_lst[0])

    return -1


  def temporal_resolution_str(self) -> str:
    interval_dict = self.stac_json.get('gee:interval')
    if not interval_dict:
      return ""
    return f"{interval_dict['interval']} {interval_dict['unit']}"


  def python_code(self)-> str:
    code = self.stac_json.get('code')
    if not code:
      return ''

    return code.get('py_code')

  def set_python_code(self, code: str):
    if not code:
      self.stac_json['code'] = {'js_code': '', 'py_code': code}

    self.stac_json['code']['py_code'] = code

  def set_js_code(self, code: str):
    if not code:
      return ''
    js_code = self.stac_json.get('code').get('js_code')
    self.stac_json['code'] = {'js_code': '', 'py_code': code}

  def image_preview_url(self):
    for link in self.stac_json['links']:
      if 'rel' in link and link['rel'] == 'preview' and link['type'] == 'image/png':
        return link['href']
    raise ValueError(f"No preview image found for {id}")


  def catalog_url(self):
    links = self.stac_json['links']
    for link in links:
      if 'rel' in link and link['rel'] == 'catalog':
        return link['href']

      # Ideally there would be a 'catalog' link but sometimes there isn't.
      base_url = "https://developers.google.com/earth-engine/datasets/catalog/"
      if link['href'].startswith(base_url):
        return link['href'].split('#')[0]

    logging.warning(f"No catalog link found for {self.public_id()}")
    return ""

