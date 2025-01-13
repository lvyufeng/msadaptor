from typing import List, Union
import os
import mindspore as ms
import torch as torch
from torch import nn, Tensor
from torch.nn.modules.utils import _pair
import logging
from mindspore.ops._primitive_cache import _get_cache_prim

from ._utils import convert_boxes_to_roi_format, check_roi_boxes_shape


def roi_align(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size,
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    """
    Performs Region of Interest (RoI) Align operator with average pooling, as described in Mask R-CNN.

    Args:
        input (Tensor[N, C, H, W]): The input tensor, i.e. a batch with ``N`` elements. Each element
            contains ``C`` feature maps of dimensions ``H x W``.
            If the tensor is quantized, we expect a batch size of ``N == 1``.
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from.
            The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            If a single Tensor is passed, then the first column should
            contain the index of the corresponding element in the batch, i.e. a number in ``[0, N - 1]``.
            If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i
            in the batch.
        output_size (int or Tuple[int, int]): the size of the output (in bins or pixels) after the pooling
            is performed, as (height, width).
        spatial_scale (float): a scaling factor that maps the box coordinates to
            the input coordinates. For example, if your boxes are defined on the scale
            of a 224x224 image and your input is a 112x112 feature map (resulting from a 0.5x scaling of
            the original image), you'll want to set this to 0.5. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly ``sampling_ratio x sampling_ratio`` sampling points per bin are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ``ceil(roi_width / output_width)``, and likewise for height). Default: -1
        aligned (bool): If False, use the legacy implementation.
            If True, pixel shift the box coordinates it by -0.5 for a better alignment with the two
            neighboring pixel indices. This version is used in Detectron2

    Returns:
        Tensor[K, C, output_size[0], output_size[1]]: The pooled RoIs.
    """

    check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)

    roi_end_mode = 0
    if aligned:
        roi_end_mode = 1
        logging.warning(
            "The implementation of 'roi_align' is not completely consistent with torch when 'aligned' is " \
            "set to True. If you are training the network, you can ignore this alarm, and if you are comparing " \
            "the accuracy of the results, please configure both parties to False. Please refer " \
            "to https://www.mindspore.cn/docs/zh-CN/r2.1/note/api_mapping/pytorch_diff/roi_align.html for " \
            "more detail."
        )

    if sampling_ratio <= 0:
        # For a batch, the number of sampling_ratio corresponding to each index may be different, cannot be
        # calculated uniformly by ``ceil(roi_width / output_width)``.
        raise NotImplementedError("sampling_ratio is not supported negative number.")

    roi_align = _get_cache_prim(ms.ops.ROIAlign)(output_size[0], output_size[1], float(spatial_scale), sampling_ratio, roi_end_mode)

    output = roi_align(input, rois)
    return output


class RoIAlign(nn.Module):
    """
    See :func:`roi_align`.
    """

    def __init__(
        self,
        output_size,
        spatial_scale: float,
        sampling_ratio: int,
        aligned: bool = False,
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        return roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"output_size={self.output_size}"
            f", spatial_scale={self.spatial_scale}"
            f", sampling_ratio={self.sampling_ratio}"
            f", aligned={self.aligned}"
            f")"
        )
        return s
