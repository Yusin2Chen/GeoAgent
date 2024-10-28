from terratorch.cli_tools import LightningInferenceModel
from huggingface_hub import hf_hub_download
from .utils.plotting import plot_rgb_agb_gedi
import subprocess

from .model_registery import ModelRegistry
terratorch_registry = ModelRegistry()

@terratorch_registry.add()
class bioMass:
    def __init__(self, ):
        self.sensor = 'Sentinel-2'
        self.task_type = 'Segmentation'

    @staticmethod
    def get_bioMass(input_dir: str = None, input_label: str = None):
        """
        Get the instance segmentation without the given prompt
        Example: samGeo.get_autoMask(model_type='vit-h')
        :param model_type: the size of the pretrained model, can be vit_h, vit_b, vit_l, vit_tiny
        :return: none
        """
        #https://colab.research.google.com/github/ibm-granite/granite-geospatial-biomass/blob/main/notebooks/agb_getting_started.ipynb#scrollTo=QtjZgw0RRjnd
        # Path to weight of model
        ckpt_path = hf_hub_download(repo_id="ibm-granite/granite-geospatial-biomass", filename="biomass_model.ckpt")
        # Path to configuration file which contains all hyperparameters
        config_path = hf_hub_download(repo_id="ibm-granite/granite-geospatial-biomass", filename="config.yaml")
        # Path to directory with geotiff test images
        if input_dir is None:
            input_dir = './test_images/inference_images'
            input_label = './test_images/inference_labels'

        # HLS bands: BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2
        # The line below names all the bands in the input, so the ones above can be extracted. we use -1 for placeholders, as we dont care about those
        UNUSED_BAND = "-1"
        predict_dataset_bands = [UNUSED_BAND, "BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2", UNUSED_BAND,
                                 UNUSED_BAND, UNUSED_BAND, UNUSED_BAND]
        model = LightningInferenceModel.from_config(config_path, ckpt_path, predict_dataset_bands)
        inference_results, input_file_names = model.inference_on_dir(input_dir)
        biome = 'Temperate Conifer Forests'
        tile_id = 'T10SFF_144'  # Located in CA, USA
        plot_rgb_agb_gedi(tile_id, input_dir, input_file_names, inference_results, input_label, biome)


@terratorch_registry.add()
class forestNet:
    def __init__(self, ):
        self.sensor = 'Sentinel-2'
        self.task_type = 'Classification'

    @staticmethod
    def get_forest_class(input_dir: str = None, output_dir: str = None):
        """
        Get the instance segmentation without the given prompt
        Example: samGeo.get_autoMask(model_type='vit-h')
        :param model_type: the size of the pretrained model, can be vit_h, vit_b, vit_l, vit_tiny
        :return: none
        """
        # 定义文件路径
        config_path = "/path/to/config/file.yaml"
        ckpt_path = "/path/to/checkpoint.ckpt"
        bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

        # 构建命令列表
        command = [
            "terratorch", "predict",
            "--config", config_path,
            "--ckpt_path", ckpt_path,
            "--predict_output_dir", output_dir,
            "--data.init_args.predict_data_root", input_dir,
            "--data.init_args.predict_dataset_bands", str(bands)
        ]

        # 运行命令并捕获输出
        result = subprocess.run(command, capture_output=True, text=True)

        # 检查执行结果
        if result.returncode == 0:
            print("命令执行成功:", result.stdout)
        else:
            print("命令执行失败:", result.stderr)


@terratorch_registry.add()
class sen2Flood:
    def __init__(self, ):
        self.sensor = 'Sentinel-2'
        self.task_type = 'Segmentation'

    @staticmethod
    def get_sen2Flood(input_dir: str = None, output_dir: str = None):
        """
        Get the instance segmentation without the given prompt
        Example: samGeo.get_autoMask(model_type='vit-h')
        :param model_type: the size of the pretrained model, can be vit_h, vit_b, vit_l, vit_tiny
        :return: none
        """
        # 定义文件路径
        config_path = "/path/to/config/file.yaml"
        ckpt_path = "/path/to/checkpoint.ckpt"
        bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

        # 构建命令列表
        command = [
            "terratorch", "predict",
            "--config", config_path,
            "--ckpt_path", ckpt_path,
            "--predict_output_dir", output_dir,
            "--data.init_args.predict_data_root", input_dir,
            "--data.init_args.predict_dataset_bands", str(bands)
        ]

        # 运行命令并捕获输出
        result = subprocess.run(command, capture_output=True, text=True)

        # 检查执行结果
        if result.returncode == 0:
            print("命令执行成功:", result.stdout)
        else:
            print("命令执行失败:", result.stderr)


@terratorch_registry.add()
class burnScar:
    def __init__(self, ):
        self.sensor = 'Sentinel-2'
        self.task_type = 'Segmentation'

    @staticmethod
    def get_burnScar(input_dir: str = None, output_dir: str = None):
        """
        Get the instance segmentation without the given prompt
        Example: samGeo.get_autoMask(model_type='vit-h')
        :param model_type: the size of the pretrained model, can be vit_h, vit_b, vit_l, vit_tiny
        :return: none
        """
        # 定义文件路径
        config_path = "/path/to/config/file.yaml"
        ckpt_path = "/path/to/checkpoint.ckpt"
        bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

        # 构建命令列表
        command = [
            "terratorch", "predict",
            "--config", config_path,
            "--ckpt_path", ckpt_path,
            "--predict_output_dir", output_dir,
            "--data.init_args.predict_data_root", input_dir,
            "--data.init_args.predict_dataset_bands", str(bands)
        ]

        # 运行命令并捕获输出
        result = subprocess.run(command, capture_output=True, text=True)

        # 检查执行结果
        if result.returncode == 0:
            print("命令执行成功:", result.stdout)
        else:
            print("命令执行失败:", result.stderr)


@terratorch_registry.add()
class multiCrop:
    def __init__(self, ):
        self.sensor = 'Sentinel-2'
        self.task_type = 'Segmentation'

    @staticmethod
    def get_multiCrop(input_dir: str = None, output_dir: str = None):
        """
        Get the instance segmentation without the given prompt
        Example: samGeo.get_autoMask(model_type='vit-h')
        :param model_type: the size of the pretrained model, can be vit_h, vit_b, vit_l, vit_tiny
        :return: none
        """
        # 定义文件路径
        config_path = "/path/to/config/file.yaml"
        ckpt_path = "/path/to/checkpoint.ckpt"
        bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

        # 构建命令列表
        command = [
            "terratorch", "predict",
            "--config", config_path,
            "--ckpt_path", ckpt_path,
            "--predict_output_dir", output_dir,
            "--data.init_args.predict_data_root", input_dir,
            "--data.init_args.predict_dataset_bands", str(bands)
        ]

        # 运行命令并捕获输出
        result = subprocess.run(command, capture_output=True, text=True)

        # 检查执行结果
        if result.returncode == 0:
            print("命令执行成功:", result.stdout)
        else:
            print("命令执行失败:", result.stderr)