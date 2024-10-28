from typing import Protocol, TypeVar, Generic, Sequence, Dict, Optional, List, Tuple, Any, Set, Union, Callable
from .gee_utils import BBox, Collection
import iso8601
#import ee
import json
import requests
from datetime import datetime
#ee.Initialize()
#import geemap
#Map = geemap.Map()
#Map.add("layer_manager")
from .data_registery import DataRegistry
#from GeoAgent.utils import s1_polarization
#from GeoAgent.utils import s1_instrumentMode

geeData_registery = DataRegistry()


@geeData_registery.add()
class S1_GRD:
    def __init__(self,):
        self.sensor = 'Sentinel-1'
        # Fetch the data
        response = requests.get("https://storage.googleapis.com/earthengine-stac/catalog/COPERNICUS/COPERNICUS_S1_GRD.json")
        # Check if the request was successful
        if response.status_code == 200:
            # Load the JSON content
            S1_prop = Collection(json.loads(response.text))
            self.bBox = S1_prop.bbox_list()
            self.timeInterval = S1_prop.datetime_interval()
        else:
            bBox = [[-180, -90, 180, 90]]
            self.bBox = tuple([BBox.from_list(x) for x in bBox])
            self.timeInterval = iter(tuple([(iso8601.parse_date("2014-10-03T00:00:00Z"), iso8601.parse_date("2024-03-13T18:04:55Z"))]))


    @staticmethod
    def get_S1_GRD(polarization: str = '', instrumentmode: str = ''):
        """
        The Sentinel-1 mission provides data from a dual-polarization C-band Synthetic Aperture Radar (SAR) instrument
        Example: img = S1_GRD.get_S1_GRD(polarization='VV', instrumentmode='IW')
        :param polarization: the polarization model (VV or VH) of Sentinel-1 images
        :param instrumentmode: the Sentinel-1 SAR imaging model
        :return: ee.ImageCollection: img
        """

        def function_S1_GRD(image):
            edge = image.lt(-30.0)
            maskedImage = image.mask().And(edge.Not())
            return image.updateMask(maskedImage)

        img = ee.ImageCollection('COPERNICUS/S1_GRD')\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization))\
            .filter(ee.Filter.eq('instrumentMode', instrumentmode))\
            .select(polarization)\
            .map(function_S1_GRD)
        return img


@geeData_registery.add()
class S2:
    def __init__(self, ):
        self.sensor = 'Sentinel-2'
        # Fetch the data
        response = requests.get("https://storage.googleapis.com/earthengine-stac/catalog/COPERNICUS/COPERNICUS_S2.json")
        # Check if the request was successful
        if response.status_code == 200:
            # Load the JSON content
            S2_prop = Collection(json.loads(response.text))
            self.bBox = S2_prop.bbox_list()
            self.timeInterval = S2_prop.datetime_interval()
        else:
            self.bBox = None
            self.timeInterval = None

    @staticmethod
    def get_COPERNICUS_S2(start_date: datetime, end_date: datetime, cloud: float):
        """
        Sentinel-2 is a wide-swath, high-resolution, multi-spectral imaging mission supporting Copernicus Land Monitoring studies,
        including the monitoring of vegetation, soil and water cover, as well as observation of inland waterways and coastal areas.
        The Sentinel-2 data contain 13 UINT16 spectral bands representing TOA reflectance scaled by 10000.
        Example: img = S2.get_COPERNICUS_S2(start_date=datetime(2020, 5, 17).replace(tzinfo=timezone.utc),
        end_date=datetime(2024, 5, 17).replace(tzinfo=timezone.utc), cloud=0.1)
        :param start_date: the start date of data query
        :param end_date: the end date of date query
        :param cloud: the probability of cloud cover
        :return: ee.ImageCollection: img
        """

        def maskS2clouds(image):
            # Bits 10 and 11 are clouds and cirrus, respectively.
            # Both flags should be set to zero, indicating clear conditions.
            qa = image.select('QA60')
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11
            mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            return image.updateMask(mask).divide(10000)

        # Map the function over one year of data and take the median.
        # Load Sentinel-2 TOA reflectance data.
        # Pre-filter to get less cloudy granules.
        dataset = ee.ImageCollection('COPERNICUS/S2') \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud)) \
            .map(maskS2clouds)
        return dataset

