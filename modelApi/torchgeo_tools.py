from typing import Protocol, TypeVar, Generic, Sequence, Dict, Optional, List, Tuple, Any, Set, Union, Callable
from torch import Tensor
import torchgeo
from model_registery import ModelRegistry
torchGeo_registry = ModelRegistry()

@torchGeo_registry.add()
class FCSiamConc:
    def __init__(self, ):
        self.sensor = 'RGB'
        self.task_type = 'Change Detection'

    @staticmethod
    def get_FCSiamConc(x: Tensor) -> Tensor:
        """
        change detection model: Fully-convolutional Siamese Concatenation (FC-Siam-conc)
        Example: mask = FCSiamConc.get_FCSiamConc(x=img)
        :param x: input images of shape (b, t, c, h, w)
        :return: predicted change masks of size (b, classes, h, w)
        """
        encoder_name: str = 'resnet34'
        encoder_depth: int = 5
        encoder_weights: str = 'imagenet'
        decoder_use_batchnorm: bool = True
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16)
        decoder_attention_type: Any = None,
        in_channels: int = 3
        classes: int = 1
        activation: Any = None
        model = torchgeo.FCSiamConc(encoder_name, encoder_depth, encoder_weights, decoder_use_batchnorm,
                                    decoder_channels, decoder_attention_type, in_channels, classes, activation)
        return model(x)


@torchGeo_registry.add()
class FCSiamDiff:
    def __init__(self, ):
        self.sensor = 'RGB'
        self.task_type = 'Change Detection'

    @staticmethod
    def get_FCSiamDiff(x: Tensor) -> Tensor:
        """
        change detection model: Fully-convolutional Siamese Difference (FC-Siam-diff)
        Example: mask = FCSiamDiff.get_FCSiamDiff(x=img)
        :param x: input images of shape (b, t, c, h, w)
        :return: predicted change masks of size (b, classes, h, w)
        """
        model = torchgeo.FCSiamDiff()
        return model(x)


@torchGeo_registry.add()
class ChangeStarFarSeg:
    def __init__(self, ):
        self.sensor = 'RGB'
        self.task_type = 'Change Detection'

    @staticmethod
    def get_ChangeStarFarSeg(x: Tensor) -> dict[str, Tensor]:
        """
        change detection model: The base class of the network architecture of ChangeStar
        Example: mask = ChangeStarFarSeg.get_ChangeStarFarSeg(x=img)
        :param x: a bitemporal input tensor of shape [B, T, C, H, W]
        :return: a dictionary containing bitemporal semantic segmentation logit and binary change detection logit/probability
        """
        model = torchgeo.ChangeStarFarSeg()
        return model(x)


@torchGeo_registry.add()
class FCN:
    def __init__(self, ):
        self.sensor = 'RGB'
        self.task_type = 'Segmentation'

    @staticmethod
    def get_FCN(x: Tensor) -> Tensor:
        """
        segmentation model: FCN model
        Example: segs = FCN.get_FCN(x=img)
        :param x: a single temporal input tensor of shape [B, C, H, W]
        :return: a dictionary containing semantic segmentation logit
        """
        in_channels: int = 3
        classes: int = 3
        num_filters: int = 64
        model = torchgeo.FCN(in_channels, classes, num_filters)
        return model(x)


@torchGeo_registry.add()
class RCF:
    def __init__(self, ):
        self.sensor = 'RGB'
        self.task_type = 'Segmentation'

    @staticmethod
    def get_RCF(x: Tensor) -> Tensor:
        """
        This model extracts random convolutional features (RCFs) from its input.
        Example: segs = RCF.get_RCF(x=img)
        :param x: a tensor with shape (B, C, H, W)
        :return: a tensor of size (B, ``self.num_features``)
        """
        in_channels: int = 4
        features: int = 16
        kernel_size: int = 3
        bias: float = -1.0
        seed = None
        mode: str = 'gaussian'
        dataset = None
        model = torchgeo.RCF(in_channels, features, kernel_size, bias)
        return model(x)


