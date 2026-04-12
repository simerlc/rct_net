# Environment Setup：

```
conda create -n rct_net python=3.7 -y
conda activate rct_net
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch -y
pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2
pip install pointnet2_ops_lib/.
```



# Download Data:

**Download the "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip" dataset file of shapenet and extract it to the "data“folder**

# Train：

```
python main.py --model rct_net
```

pretrained model:
链接: https://pan.baidu.com/s/1AACLUoF0OzWM9trT9vG0hg?pwd=98jd 提取码: 98jd
