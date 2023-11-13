# NTED Step Generation

The PyTorch implementation based on paper NTED

* [Paper](Neural Texture Extraction and Distribution for  Controllable Person Image Synthesis) (**CVPR2022 Oral**)
* [Github](https://github.com/RenYurui/Neural-Texture-Extraction-Distribution)



<p align='center'>  
  <img src='https://user-images.githubusercontent.com/26128046/282382834-bf19fbb3-9085-42b1-b4e7-68be354c7231.png' width='2000'/>
</p>

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/26128046/282382398-a822b36b-f54f-4aff-8057-4d455eb6fd64.png' width='2000'/>
</p>



## Generate Image Step-by-Step

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/26128046/282384506-83b9795d-0e62-448a-9347-ffff81e62663.png' width='2000'/>
</p>

This section describes the process of generating ground-truth images by downsampling and then upsampling back to the original size.

#### Downsampling Process

1. **Resolution Range**: The downsampling resolutions are linearly constructed, ranging from `[8, 5]` to `[256, 176]`.
2. **Steps**: The resolution adjustment is carried out in 20 steps.

#### Training Scheme

1. **Random Timestep Selection**: During the training process, 3 random timesteps are selected.

This methodology is crucial for preparing the dataset and assessing the image restoration capabilities of the model. By undergoing the downsampling and upsampling processes, the model's ability to maintain and restore the characteristics of the original image can be evaluated.

### Step Prediction (ACGAN)

Building upon the framework of **[Auxiliary Classifier GAN (ACGAN)](https://proceedings.mlr.press/v70/odena17a.html) (PMLR 2017)**, we have incorporated an additional training process: the step prediction loss. This enhancement aims to stabilize the training phase.

#### Key Improvement:

- **Step Prediction Loss**: This added component mitigates a common issue in the original ACGAN framework. Without step prediction, the Discriminator often struggles to accurately determine which step of image generation should be executed. This confusion can lead to the generation of artifacts, such as image cracking. By integrating step prediction loss into the training process, we enhance the Discriminator's ability to more accurately guide the generation process, thereby reducing the occurrence of artifacts.

This modification to the ACGAN framework not only stabilizes the training process but also improves the overall quality of the generated images.

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/26128046/282387633-0e9c8b01-d0b7-4a80-bd1a-35ee52dd9a8d.png' width='2000'/>
</p>

## Installation

#### Requirements

- Python 3
- PyTorch 1.7.1
- CUDA 10.2

#### Conda Installation

``` bash
# 1. Create a conda virtual environment.
conda create -n NTED_step python=3.8
conda activate NTED_step
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.3

# 2. Clone the Repo and Install dependencies
git clone --recursive https://github.com/RenYurui/Neural-Texture-Extraction-Distribution.git
pip install -r requirements.txt

# 3. Install mmfashion (for appearance control only)
pip install mmcv==0.5.1
pip install pycocotools==2.0.4
cd ./scripts
chmod +x insert_mmfashion2mmdetection.sh
./insert_mmfashion2mmdetection.sh
cd ../third_part/mmdetection
pip install -v -e .
```



## Dataset

- Download `img_highres.zip` of the DeepFashion Dataset from [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00). 

- Unzip `img_highres.zip`. You will need to ask for password from the [dataset maintainers](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Then rename the obtained folder as **img** and put it under the `./dataset/deepfashion` directory. 

- We split the train/test set following [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention). Several images with significant occlusions are removed from the training set. Download the train/test pairs and the keypoints `pose.zip` extracted with [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) by runing: 

  ```bash
  cd scripts
  ./download_dataset.sh
  ```

  Or you can download these files manually：

  - Download the train/test pairs from [Google Drive](https://drive.google.com/drive/folders/1PhnaFNg9zxMZM-ccJAzLIt2iqWFRzXSw?usp=sharing) including **train_pairs.txt**, **test_pairs.txt**, **train.lst**, **test.lst**. Put these files under the  `./dataset/deepfashion` directory. 
  - Download the keypoints `pose.rar` extracted with Openpose from [Google Driven](https://drive.google.com/file/d/1waNzq-deGBKATXMU9JzMDWdGsF4YkcW_/view?usp=sharing). Unzip and put the obtained floder under the  `./dataset/deepfashion` directory.

- Run the following code to save images to lmdb dataset.

  ```bash
  python -m scripts.prepare_data \
  --root ./dataset/deepfashion \
  --out ./dataset/deepfashion
  ```



## Training 

This project supports multi-GPUs training. The following code shows an example for training the model with 256x176 images using 4 GPUs.

  ```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port 1234 train.py \
--config ./config/fashion_256.yaml \
--name $name_of_your_experiment
  ```

All configs for this experiment are saved in `./config/fashion_256.yaml`. 
If you change the number of GPUs, you may need to modify the `batch_size` in `./config/fashion_256.yaml` to ensure using a same `batch_size`.



## Inference

- **Download the trained weights for [512x352 images](https://drive.google.com/file/d/1eM2ikE2o0T5376rAV5nrTNjDE4Rh18_a/view?usp=sharing) and [256x176 images](https://drive.google.com/file/d/1CnXLtpTGSKHMeOyyjd5GkaMVIF2eBtkz/view?usp=sharing)**. Put the obtained checkpoints under `./result/fashion_512` and `./result/fashion_256` respectively.

- Run the following code to evaluate the trained model:

  ```bash
  python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port 12345 inference.py \
  --config ./config/fashion_256.yaml \
  --name fashion_256 \
  --no_resume \
  --output_dir ./result/fashion_256/inference 
  ```



## Evaluation
