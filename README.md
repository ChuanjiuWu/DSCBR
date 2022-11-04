# Dual-Supervised Contrastive Learning for Bundle Recommendation
This is our Pytorch implementation for the paper:





## Requirements

* python == 3.8.12 
* supported(tested) CUDA versions: 10.1
* Pytorch == 1.4.0 or above


## Code Structure
1. The entry script for training and evaluation is: [train.py]
2. The config file is: [config.yaml]
3. The script for data preprocess and dataloader: [utility.py]
4. The model folder: [./models]
5. The experimental logs in tensorboard-format are saved in [./runs.]
6. The experimental logs in txt-format are saved in [./log.]
7. The best model and associate config file for each experimental setting is saved in [./checkpoints.]

## How to run the code
1. Decompress the dataset file into the current folder: 

   > tar -zxvf dataset.tgz
 
   Noted: for the iFashion dataset, we incorporate three additional files: user\_id\_map.json, item\_id\_map.json, and bundle\_id\_map.json, which record the id mappings between the original string-formatted id in the POG dataset and the integer-formatted id in our dataset. You may use the mappings to obtain the original content information of the items/outfits. We do not use any content information in this work.

2. Train CrossCBR on the dataset Youshu with GPU 0: 

   > python train.py -g 0 -m DSCBR -d Youshu

   You can specify the gpu id and the used dataset by cmd line arguments, while you can tune the hyper-parameters by revising the configy file [config.yaml]. The detailed introduction of the hyper-parameters can be seen in the config file, and you are highly encouraged to read the paper to better understand the effects of some key hyper-parameters.
   
   
   ##over
