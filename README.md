# Fourier Filtering Traffic Low-Light Segmentation(ITSC2025)
## 环境准备
### conda环境
```
conda create --name your_env
conda activate your_env
conda install python==3.8
```
### 依赖安装
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch 
pip install -U openmim
mim install mmcv==2.0.1
mim install mmdet==3.3.0
mim install mmengine==0.10.3
mim install mmyolo==0.6.0
pip install supervision
pip install transformers
pip install ninja tqdm
```
### Third-Part准备
```
# 准备mmyolo
cd third_part
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
pip install -r requirements/albu.txt
mim install -v -e .
# 准备clip预训练模型
git clone https://huggingface.co/openai/clip-vit-base-patch32
```
## 数据集准备
将数据集下载至`F-Seg/datasets`文件夹：
["NightCity"](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html)
## 训练
```
cd scripts
python train_edge.py
```
## 测试demo
```
cd scripts
python demo.py
```