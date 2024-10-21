from .dpt import (
    DPTDepthModel, DPTSegmentationModel,
    BiBertDPTDepthModel, BiBertDPTSegmentationModel,
    BiTDPTDepthModel, BiTDPTSegmentationModel,
    BiViTDPTDepthModel, BiViTDPTSegmentationModel,
    BiDenseDPTDepthModel, BiDenseDPTSegmentationModel,
)
from .upernet import (
    UperNetDepthModel, UperNetSegmentationModel,
    BnnUperNetDepthModel, BnnUperNetSegmentationModel,
    ReActUperNetDepthModel, ReActUperNetSegmentationModel,
    AdaBinUperNetDepthModel, AdaBinUperNetSegmentationModel,
    BiSRUperNetDepthModel, BiSRUperNetSegmentationModel,
    CFBUperNetDepthModel, CFBUperNetSegmentationModel,
    BiDenseUperNetDepthModel, BiDenseUperNetSegmentationModel
)


DEPTH_MODEL_DICT = {
    'dpt': {
        'fp32': DPTDepthModel,
        'bibert': BiBertDPTDepthModel,
        'bit': BiTDPTDepthModel,
        'bivit': BiViTDPTDepthModel,
        'bidense': BiDenseDPTDepthModel,
    },

    'upernet': {
        'fp32': UperNetDepthModel,
        'bnn': BnnUperNetDepthModel,
        'react': ReActUperNetDepthModel,
        'adabin': AdaBinUperNetDepthModel,
        'bisrnet': BiSRUperNetDepthModel,
        'cfb': CFBUperNetDepthModel,
        'bidense': BiDenseUperNetDepthModel,
    },
}


SEGMENTATION_MODEL_DICT = {
    'dpt': {
        'fp32': DPTSegmentationModel,
        'bibert': BiBertDPTSegmentationModel,
        'bit': BiTDPTSegmentationModel,
        'bivit': BiViTDPTSegmentationModel,
        'bidense': BiDenseDPTSegmentationModel,
    },

    'upernet': {
        'fp32': UperNetSegmentationModel,
        'bnn': BnnUperNetSegmentationModel,
        'react': ReActUperNetSegmentationModel,
        'adabin': AdaBinUperNetSegmentationModel,
        'bisrnet': BiSRUperNetSegmentationModel,
        'cfb': CFBUperNetSegmentationModel,
        'bidense': BiDenseUperNetSegmentationModel,
    },
}