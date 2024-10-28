from autodistill_efficient_yolo_world import EfficientYOLOWorld
from autodistill.detection import CaptionOntology
import cv2
import supervision as sv
from .model_registery import ModelRegistry
yolo_registry = ModelRegistry()
'''
!pip install autodistill-efficient-yolo-world
!pip install roboflow
!mkdir -p /root/.autodistill/
!wget https://huggingface.co/spaces/SkalskiP/YOLO-World/resolve/main/efficient_sam_s_cpu.jit?download=true -O /root/.autodistill/efficient_sam_s_cpu.jit
'''

@yolo_registry.add()
class yoloWorld:
    def __init__(self, ):
        self.sensor = 'RGB'
        self.task_type = 'Object Detection'

    @staticmethod
    def get_yoloWorld(input_img: str = './img.png', output_dir: str = '', obj_caption: str = 'book', obj_class: str = 'book'):
        """
        Get the instance segmentation without the given prompt
        Example: samGeo.get_autoMask(model_type='vit-h')
        :param model_type: the size of the pretrained model, can be vit_h, vit_b, vit_l, vit_tiny
        :return: none
        """
        # define an ontology to map class names to our EfficientYOLOWorld prompt
        # the ontology dictionary has the format {caption: class}
        # where caption is the prompt sent to the base model, and class is the label that will
        # be saved for that caption in the generated annotations
        # then, load the model
        base_model = EfficientYOLOWorld(ontology=CaptionOntology({obj_caption: obj_class}))
        # predict on an image
        result = base_model.predict(input_img, confidence=0.1)
        image = cv2.imread(input_img)
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(
            scene=image.copy(),
            detections=result,
        )
        sv.plot_image(annotated_frame)
        base_model.label("./context_images", extension=".jpeg")
