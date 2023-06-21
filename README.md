# MMVC: Learned Multi-Mode Video Compression with Block-based Prediction Mode Selection and Density-Adaptive Entropy Coding
Codes for [MMVC: Learned Multi-Mode Video Compression with Block-based Prediction Mode Selection and Density-Adaptive Entropy Coding](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_MMVC_Learned_Multi-Mode_Video_Compression_With_Block-Based_Prediction_Mode_Selection_CVPR_2023_paper.pdf) (CVPR 2023), a block-wised video compression algorithm.

## Introduction and Framework
In this paper, we propose multi-mode video compression (MMVC), a block-wise mode ensemble deep video compression framework that selects the optimal mode for feature domain prediction adapting to different motion patterns.

Proposed multi-modes include ConvLSTM-based feature domain prediction, optical flow conditioned feature domain prediction, and feature propagation to address a wide range of cases from static scenes without apparent motions to dynamic scenes with a moving camera. We partition the feature space into blocks for temporal prediction in spatial block-based representations.

We consider both dense and sparse post-quantization residual blocks for entropy coding, and apply optional run-length coding to sparse residuals to improve the compression rate.

![Flow chart](https://github.com/BowenL0218/MMVC_video_codec/blob/main/Images/Flowchart.png)

## Performance
### RD trade-off curve
MMVC achieves state-of-the-art performance on benchmark datasets
![Curve](https://github.com/BowenL0218/MMVC_video_codec/blob/main/Images/Curve.png)

### Reconstruction Visualization
Details of the static background and dynamic objects are well preserved. Compared with HEVC, our result yields fewer block artifacts preserving finer details.
![Vis](https://github.com/BowenL0218/MMVC_video_codec/blob/main/Images/Vis.png)

### Visualization of Multiple Predictions
The decoded scenes are obtained from the predicted features without residual. By adopting multiple prediction modes that complement each other, our prediction is able to cover content variety in the original frame with a shorter bitstream.
![Mode_sele](https://github.com/BowenL0218/MMVC_video_codec/blob/main/Images/Mode_sele.png)

## Ablation Study
### Effectiveness of Optical Flow conditioned Prediction
The residual between the raw frame and the wrapped frame with optical flow information is minor when the motion is slow, indicating that the optical flow based prediction mode works well with some static frames.

![OFC](https://github.com/BowenL0218/MMVC_video_codec/blob/main/Images/Opti_vis.png)

### Mapping between pixel domain and latent domain
The spatial correlation between the pixel domain and latent space is straightforward. The reconstructions of the divided latent are stitched into a single frame, and the difference between the reconstruction and raw frame is small, showing the effectiveness of block-wised video compression in latent space.

![Stit](https://github.com/BowenL0218/MMVC_video_codec/blob/main/Images/stitching.png)

## Datasets
In order to use the datasets used in the paper, please download the [UVG dataset](https://media.withyoutube.com/), the [Kinetics dataset](https://deepmind.com/research/open-source/kinetics), and the [UVG dataset](http://ultravideo.fi/).

- The UVG and Kinetics datasets are used for training the prediction network. 
- The Kinetics and UVG datasets are implemented for testing the performance.
- Note that we use the learning-based image compression algorithm ([Liu et al](https://arxiv.org/pdf/1912.03734.pdf)) as the intra-compression for one single frame. 
- The latent is used as the optimal latent for each frame.
  
## Arithmetic Coding
To use the entropy coding method in this paper, download the general code library in Python with [arithmetic coding](https://github.com/ahmedfgad/ArithmeticEncodingPython). (The code is not provided in this repository due to the license conflict.)

## Train a new model
Please download the optical flow model into the Optical_flow_master folder, and change the path in the corresponding file. The method that we used in this paper is RAFT algorithm. 
To train a model with optical flow conditioned prediction, 
```sh
$ python train_new_model.py
```

To test the result with the trained model
```sh
$ python Testing_new.py
```

## Citation
Please cite our paper if you find our paper useful for your research. [MMVC](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_MMVC_Learned_Multi-Mode_Video_Compression_With_Block-Based_Prediction_Mode_Selection_CVPR_2023_paper.pdf)

@inproceedings{liu2023mmvc,
  title={MMVC: Learned Multi-Mode Video Compression with Block-based Prediction Mode Selection and Density-Adaptive Entropy Coding},
  author={Liu, Bowen and Chen, Yu and Machineni, Rakesh Chowdary and Liu, Shiyu and Kim, Hun-Seok},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18487--18496},
  year={2023}
}
