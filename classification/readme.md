# Adaptive Feature Swapping


## Environments

* **Ubuntu 16.04.1**
* **CUDA 10.0.130**
* **Python 3.6.10**
* **Pytorch 1.3.1**
* **Numpy 1.19.1**
* **Scipy 1.5.2**
* **TorchVision 0.4.2**

## Dataset

To run our code, please first download the required dataset. Please change the root in Line 384 and Line 389 to the directories that your [office-home](https://www.hemanthdv.org/officeHomeDataset.html) and [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) datasets locate.

## Training

Run the script for training.

```shell
python da_afs.py --dset office-home --lr 0.005 --net resnet50 --gpu_id $GPU_DEVICE --batch_size 36 --method MCC+S

python da_afs.py --dset visda --lr 0.001 --net resnet101 --gpu_id $GPU_DEVICE --batch_size 36 --method MCC+S
```
