from .dpt import DPTDepthModel, DPTSegmentationModel
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