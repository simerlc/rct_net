# Environment Setup：

```
conda create -n rct_net python=3.7 -y
conda activate rct_net
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch -y
pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2
pip install pointnet2_ops_lib/.
```



# Download Data:

**Download the "modelnet40_ply_hdf5_2048.zip"dataset file of modelnet40 and extract it to the "data“ folder.**

**Download the "h5_files.zip" dataset file of  ScanObjectNN and extract it to the "data" folder.**

# Train：

### Classification ModelNet40：

```
python main.py --model rct_net
```



### Classification ScanObjectNN:

```
python main_scan.py --model rct_net
```





