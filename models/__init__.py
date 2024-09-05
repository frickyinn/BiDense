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
