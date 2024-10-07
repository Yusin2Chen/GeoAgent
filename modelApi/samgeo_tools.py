from typing import Protocol, TypeVar, Generic, Sequence, Dict, Optional, List, Tuple, Any, Set, Union, Callable
'''
import leafmap
try:
    import samgeo  #https://samgeo.gishub.org/
except:
    get_ipython().system('pip install samgeo')
from samgeo.hq_sam import (
    SamGeo,
    show_image,
    download_file,
    overlay_images,
    tms_to_geotiff,
)
from samgeo.text_sam import LangSAM
'''
from .model_registery import ModelRegistry

samGeo_registry = ModelRegistry()

@samGeo_registry.add()
class samGeo:
    def __init__(self, ):
        self.sensor = 'RGB'
        self.task_type = 'Segmentation'

    @staticmethod
    def get_autoMask(model_type: str = 'vit_h'):
        """
        Get the instance segmentation without the given prompt
        Example: samGeo.get_autoMask(model_type='vit-h')
        :param model_type: the size of the pretrained model, can be vit_h, vit_b, vit_l, vit_tiny
        :return: none
        """
        m = leafmap.Map(center=[37.6412, -122.1353], zoom=15, height="800px")
        m.add_basemap("SATELLITE")
        if m.user_roi is not None:
            bbox = m.user_roi_bounds()
        else:
            bbox = [-122.1497, 37.6311, -122.1203, 37.6458]
        image = "satellite.tif"
        tms_to_geotiff(output=image, bbox=bbox, zoom=17, source="Satellite", overwrite=True)
        m = leafmap.Map(center=[37.8713, -122.2580], zoom=17, height="800px")
        m.add_basemap("SATELLITE")
        m.layers[-1].visible = False
        m.add_raster(image, layer_name="Image")

        sam_kwargs = {
            "points_per_side": 32,
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 100,
        }
        sam = SamGeo(
            model_type=model_type,
            sam_kwargs=sam_kwargs,
        )
        sam.generate(image, output="masks.tif", foreground=True, unique=True)
        sam.show_masks(cmap="binary_r")
        sam.show_anns(axis="off", alpha=1, output="annotations.tif")
        leafmap.image_comparison(
            "satellite.tif",
            "annotations.tif",
            label1="Satellite Image",
            label2="Image Segmentation",
        )
        m.add_raster("annotations.tif", alpha=0.5, layer_name="Masks")
        sam.tiff_to_vector("masks.tif", "masks.gpkg")

    @staticmethod
    def get_pointMask(point_coords: list, model_type: str = 'vit_h'):
        """
        get the instance segmentation given points prompt
        Example: samGeo.get_pointMask(point_coords=[[-122.1497, 37.6311], [-122.1203, 37.6458]], model_type='vit-h')
        :param point_coords: the point prompt, e.g., List[List[float, float]]
        :param model_type: the size of the pretrained model, can be vit_h, vit_b, vit_l, vit_tiny
        :return: none
        """
        m = leafmap.Map(center=[37.6412, -122.1353], zoom=15, height="800px")
        m.add_basemap("SATELLITE")
        if m.user_roi is not None:
            bbox = m.user_roi_bounds()
        else:
            bbox = [-122.1497, 37.6311, -122.1203, 37.6458]
        image = "satellite.tif"
        tms_to_geotiff(output=image, bbox=bbox, zoom=16, source="Satellite", overwrite=True)
        m.layers[-1].visible = False
        m.add_raster(image, layer_name="Image")
        sam = SamGeo(
            model_type=model_type,
            automatic=False,
            sam_kwargs=None,
        )
        sam.set_image(image)
        sam.predict(point_coords, point_labels=1, point_crs="EPSG:4326", output="mask2.tif")
        m.add_raster("mask2.tif", layer_name="Mask2", nodata=0, cmap="Greens", opacity=1)

    @staticmethod
    def get_boxMask(model_type: str = 'vit_h'):
        """
        get the instance segmentation given boxes prompt
        Example: samGeo.get_boxMask(model_type='vit-h')
        :param model_type: the size of the pretrained model, can be vit_h, vit_b, vit_l, vit_tiny
        :return: none
        """
        m = leafmap.Map(center=[-22.17615, -51.253043], zoom=18, height="800px")
        m.add_basemap("SATELLITE")
        bbox = m.user_roi_bounds()
        if bbox is None:
            bbox = [-51.2565, -22.1777, -51.2512, -22.175]
        image = "Image.tif"
        tms_to_geotiff(output=image, bbox=bbox, zoom=19, source="Satellite", overwrite=True)
        m.layers[-1].visible = False
        m.add_raster(image, layer_name="Image")
        sam = SamGeo(
            model_type=model_type,
            automatic=False,
            sam_kwargs=None,
        )
        sam.set_image(image)
        if m.user_rois is not None:
            boxes = m.user_rois
        else:
            boxes = [
                [-51.2546, -22.1771, -51.2541, -22.1767],
                [-51.2538, -22.1764, -51.2535, -22.1761],
            ]
        sam.predict(boxes=boxes, point_crs="EPSG:4326", output="mask.tif", dtype="uint8")
        m.add_raster("mask.tif", cmap="viridis", nodata=0, layer_name="Mask")

    @staticmethod
    def get_textMask(text_prompt: str):
        """
        get the instance segmentation given text prompt
        Example: samGeo.get_textMask((text_prompt='tree')
        :param text_prompt: given the target object, e.g., tree
        :return: none
        """
        m = leafmap.Map(center=[-22.17615, -51.253043], zoom=18, height="800px")
        m.add_basemap("SATELLITE")
        bbox = m.user_roi_bounds()
        if bbox is None:
            bbox = [-51.2565, -22.1777, -51.2512, -22.175]
        image = "Image.tif"
        tms_to_geotiff(output=image, bbox=bbox, zoom=19, source="Satellite", overwrite=True)
        m.layers[-1].visible = False
        m.add_raster(image, layer_name="Image")
        sam = LangSAM()
        sam.predict(image, text_prompt, box_threshold=0.24, text_threshold=0.24)
        sam.show_anns(
            cmap="Greens",
            box_color="red",
            title="Automatic Segmentation of Trees",
            blend=True,
        )
        sam.show_anns(
            cmap="Greens",
            add_boxes=False,
            alpha=0.5,
            title="Automatic Segmentation of Trees",
        )
        sam.show_anns(
            cmap="Greys_r",
            add_boxes=False,
            alpha=1,
            title="Automatic Segmentation of Trees",
            blend=False,
            output="trees.tif",
        )
        sam.raster_to_vector("trees.tif", "trees.shp")





