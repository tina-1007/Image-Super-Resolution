# Image  Super-Resolution

In this homework, I apply [SwinIR](https://arxiv.org/abs/2108.10257) on the task [2021 VRDL HW4](https://codalab.lisn.upsaclay.fr/competitions/622?secret_key=4e06d660-cd84-429c-971b-79d15f78d400)


## Installation

```
pip install -r requirements.txt
```

## Prepare data
#### 1. Download the dataset from CodaLab

* Put two folders `training_hr_images/` and `testing_lr_images/` in `datasets/` folder.
* The structure will be like: 
  ```
  main_train_psnr.py
  ...
  datasets
  |- training_hr_images
  |- testing_lr_images
  |- generate_lr_images.py
  ```
#### 2. Generate low-resolution images and split dataset
```
cd datasets
python generate_lr_images.py
```
You can adjust the split condition in [line 21](https://github.com/tina-1007/Image-Super-Resolution/blob/0371c285a075bdab7adecc658a19a3de599404b0/datasets/generate_lr_images.py#L21) of `generate_lr_images.py`.

## Testing
#### 1. Download the trained weights 
Get my trained model from [here](https://drive.google.com/file/d/1zYvwOylTeCwhq_wGL7mvUoFQ0X7v4kxK/view?usp=sharing) and put it in `checkpoints/`

#### 2. Generate result images
``` 
python inference.py --scale 3 --model_path checkpoints/swinir_classical_sr_x3.pth --folder_lq datasets/testing_lr_images --save_dir results
```
The upscaling images will be generated in `results/`.

## Training
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_psnr.py --opt options/train_swinir_sr_classical.json  --dist True
```
* Make sure below three numbers keep same
1. **--nproc_per_node** in your argument
2. **dataloader_batch_size** in `train_swinir_sr_classical.json`
3. Numbers of GPU, you can set it by **gpu_ids** in `train_swinir_sr_classical.json`

## Reference

1. SwinIR Model - https://github.com/JingyunLiang/SwinIR
2. SwinIR Training Code - https://github.com/cszn/KAIR/blob/master/docs/README_SwinIR.md
