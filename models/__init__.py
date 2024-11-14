from .dpt import (
    # DPTDepthModel, DPTSegmentationModel,
    # BiBertDPTDepthModel, BiBertDPTSegmentationModel,
    # BiTDPTDepthModel, BiTDPTSegmentationModel,
    # BiViTDPTDepthModel, BiViTDPTSegmentationModel,
    BiDenseDPTDepthModel, BiDenseDPTSegmentationModel,
)
from .upernet import (
    # UperNetDepthModel, UperNetSegmentationModel,
    # FP32UperNetDepthModel, FP32UperNetSegmentationModel,
    # BnnUperNetDepthModel, BnnUperNetSegmentationModel,
    # ReActUperNetDepthModel, ReActUperNetSegmentationModel,
    # AdaBinUperNetDepthModel, AdaBinUperNetSegmentationModel,
    # BiSRUperNetDepthModel, BiSRUperNetSegmentationModel,
    # CFBUperNetDepthModel, CFBUperNetSegmentationModel,
    BiDenseUperNetDepthModel, BiDenseUperNetSegmentationModel,
    # TestUperNetDepthModel, TestUperNetSegmentationModel,
)


DEPTH_MODEL_DICT = {
    'dpt': {
        # 'fp32': DPTDepthModel,
        # 'bibert': BiBertDPTDepthModel,
        # 'bit': BiTDPTDepthModel,
        # 'bivit': BiViTDPTDepthModel,
        'bidense': BiDenseDPTDepthModel,
    },

    'upernet': {
        # 'fp32': UperNetDepthModel,
        # 'fp32_0': FP32UperNetDepthModel,
        # 'bnn': BnnUperNetDepthModel,
        # 'react': ReActUperNetDepthModel,
        # 'adabin': AdaBinUperNetDepthModel,
        # 'bisrnet': BiSRUperNetDepthModel,
        # 'cfb': CFBUperNetDepthModel,
        'bidense': BiDenseUperNetDepthModel,
        # 'test': TestUperNetDepthModel,
    },
}


SEGMENTATION_MODEL_DICT = {
    'dpt': {
        # 'fp32': DPTSegmentationModel,
        # 'bibert': BiBertDPTSegmentationModel,
        # 'bit': BiTDPTSegmentationModel,
        # 'bivit': BiViTDPTSegmentationModel,
        'bidense': BiDenseDPTSegmentationModel,
    },

    'upernet': {
        # 'fp32': UperNetSegmentationModel,
        # 'fp32_0': FP32UperNetSegmentationModel,
        # 'bnn': BnnUperNetSegmentationModel,
        # 'react': ReActUperNetSegmentationModel,
        # 'adabin': AdaBinUperNetSegmentationModel,
        # 'bisrnet': BiSRUperNetSegmentationModel,
        # 'cfb': CFBUperNetSegmentationModel,
        'bidense': BiDenseUperNetSegmentationModel,
        # 'test': TestUperNetSegmentationModel,
    },
}