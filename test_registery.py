from datetime import datetime, timezone
from dataApi.gee_utils import BBox
from dataApi.geeAlldata_tool import geeData_registery
from modelApi.samgeo_tools import samGeo_registry

print('data modules')
print(geeData_registery.functions)
test_geeData = geeData_registery.to_list_infos(query_bbox=BBox(2.3358203, 48.8421609, 2.3709914, 48.8624786),
                                               query_interval=(datetime(2020, 5, 17).replace(tzinfo=timezone.utc), datetime(2024, 5, 17).replace(tzinfo=timezone.utc)),
                                               sensor=None)
print('gee test2', test_geeData)
#import inspect
#print(inspect.getsource(test_geeData[0][1]))

print('model modules')
print(samGeo_registry.functions)
test_sam = samGeo_registry.to_list_infos(sensor='RGB', task_type='Change Detection')
print('sam', test_sam)
