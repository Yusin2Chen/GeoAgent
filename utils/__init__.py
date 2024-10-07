from pydantic import BaseModel
from typing import Protocol, TypeVar, Generic, Sequence, Dict, Optional, List, Tuple, Any, Set, Union, Callable


class s1_polarization(BaseModel):
    """
    Desp: the polarization of sentinel-1 images should be one of ['VV', 'VH']
    Exmp: polarization = s1_polarization(polarization=VV)
    """
    polarization: str


class s1_instrumentMode(BaseModel):
    """
    Desp: the instrumentMode of sentinel-1 images should be one of ['IW', 'ST']
    Exmp: instrumentMode = s1_instrumentMode(instrumentMode='IW')
    """
    instrumentMode: str



