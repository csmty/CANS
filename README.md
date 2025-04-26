
# Rethinking Reconstruction and Denoising in the Dark:New Perspective, General Architecture and Beyond
## :star:Accepted by CVPR 2025
This is a Tensorflow implementation of Rethinking Reconstruction and Denoising in the Dark:New Perspective, General Architecture and Beyond in CVPR 2025, by Tengyu Ma, Long Ma, Ziye Li,Yuetong Wang,Jinyuan Liu,Chengpei Xu, and Risheng Liu.
![Comparison among recent state-of-the-art methods and our method.](\Figs\first.png)
Comparison among recent state-of-the-art methods and our method.


## 🚩Abstract
Recently, enhancing image quality in the original RAW domain has garnered significant attention, with denoising and reconstruction emerging as fundamental tasks. Although some works attempt to couple these tasks, they primarily focus on multi-stage learning while neglecting task associativity within a broader parameter space, leading to suboptimal performance. This work introduces a novel approach by rethinking denoising and reconstruction from a “backbone-head” perspective, leveraging the stronger shared parameter space offered by the backbone, compared to the encoder used in existing works. We derive task specific heads with fewer parameters to mitigate learning pressure. By incorporating chromaticity-aware attention into the backbone and introducing an adaptive denoising prior during training, we enable simultaneous reconstruction and denoising. Additionally, we design a dual-head interaction module to capture the latent correspondence between the two tasks, significantly enhancing multi-task accuracy. Extensive experiments validate the superiority of the proposed method.




## 😊Requirements
- python3.7
- pytorch==1.8.0
- cuda11.1
## 🔧Dataset used in the paper
SID(including Sony subset and Fuji subset):This paper randomly selected -- image pairs and -- image pairs from the Sony dataset,and -- image pairs and -- image pairs from the Fuji dataset for training and testing respetively.You can download it directly from Google drive for the [Sony](https://storage.googleapis.com/isl-datasets/SID/Sony.zip) (25 GB) and [Fuji](https://storage.googleapis.com/isl-datasets/SID/Fuji.zip)(52 GB) sets.

MCR:This paper randomly selected -- image pairs and -- image pairs from the MCR dataset for training and testing respetively.
You can download it directly from Google drive for the [MCR](https://drive.google.com/file/d/1Q3NYGyByNnEKt_mREzD2qw9L2TuxCV_r/view?usp=share_link)  sets.

LRD:This paper randomly selected -- image pairs and -- image pairs from the LRD dataset for training and testing respetively.
You can download it directly from Google drive for the [LRD](https://www.dropbox.com/scl/fi/tari9hvf2ubxo232owekz/LRD.zip?rlkey=7zj9kqgsf06g97smexkrzv05q&dl=0) sets.
## 🎬Training & Testing
The pre-trained model can be found at #model_best.pth.

You can change the input dataset path and the output path by modifying the paths in the CDCR_interact/base.yaml file. 

You can perform training and testing through the following commands:runner.py, runner_test.py. 



## 🔥Comparison results of the reconstruction task
![Comparison results of the reconstruction task.](\Figs\com_2.png)
## 🔥Comparison results of the denoising task
![Comparison results of the denoising task.](\Figs\com_denoising.png)
## ⚡Results on High-level Vision Tasks
![Comparison results of the denoising task.](\Figs\depth_segment.png)
## 📏Citation
If you use our code and dataset for research, please cite our paper:

Tengyu Ma, Long Ma, Ziye Li,Yuetong Wang,Jinyuan Liu,Chengpei Xu, and Risheng Liu,"Rethinking Reconstruction and Denoising in the Dark:New Perspective, General Architecture and Beyond",in CVPR, 2025.
