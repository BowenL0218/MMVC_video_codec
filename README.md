# MMVC: Learned Multi-Mode Video Compression with Block-based Prediction Mode Selection and Density-Adaptive Entropy Coding
Codes for [MMVC: Learned Multi-Mode Video Compression with Block-based Prediction Mode Selection and Density-Adaptive Entropy Coding](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_MMVC_Learned_Multi-Mode_Video_Compression_With_Block-Based_Prediction_Mode_Selection_CVPR_2023_paper.pdf) (CVPR 2023), a block-wised video compression algorithm.

# Introduction and Framework
In this paper, we propose multi-mode video compression (MMVC), a block-wise mode ensemble deep video compression framework that selects the optimal mode for feature domain prediction adapting to different motion patterns.

Proposed multi-modes include ConvLSTM-based feature domain prediction, optical flow conditioned feature domain prediction, and feature propagation to address a wide range of cases from static scenes without apparent motions to dynamic scenes with a moving camera. We partition the feature space into blocks for temporal prediction in spatial block-based representations.

We consider both dense and sparse post-quantization residual blocks for entropy coding, and apply optional run-length coding to sparse residuals to improve the compression rate.

![Flow chart](https://github.com/BowenL0218/MMVC_video_codec/blob/main/Images/Flowchart.png)
