# BiDense: Binarization for Dense Prediction

Dense prediction is one of the most essential tasks in computer vision. However, previous methods often require extensive computational resources, which hinders their real-world application. In this paper, we propose BiDense, a generalized binarization framework designed for efficient and accurate dense prediction tasks. BiDense incorporates two key techniques: the Distribution-adaptive Binarizer (DAB) and the Channel-adaptive Full-precision Bypass (CFB).
The DAB adaptively calculates thresholds and scaling factors for binarization, effectively retaining more information within binary neural networks. 
Meanwhile, the CFB facilitates full-precision bypassing for binary convolutional layers undergoing various channel size transformations, which enhances the propagation of real-valued signals and minimizes information loss.
By employing these techniques, BiDense preserves more real-valued information, enabling more accurate and detailed dense predictions. 
Extensive experiments demonstrate that our framework achieves superior performance compared to previous binary neural networks, closely approaching the performance levels of full-precision models.
