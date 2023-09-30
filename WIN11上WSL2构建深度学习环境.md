# 前言
显卡驱动+CUDA工具包+cuDNN 是必须品，一般根据情况来装
  
# 装子系统
1.在系统设置里面打开子系统和虚拟机设置  
2.wsl --install  
（看情况运行wsl --set-default-version 2）  
（去微软商店安装ubuntu wsl -l -v看wsl有的环境和版本）  
  
# 装CUDA配件
## 装CUDA驱动
`nvidia-smi`看看有没有驱动，没有就去给win11系统装驱动，有驱动的话看看cuda版本是多少，这决定之后CUDA toolkit和cudnn怎么装
## 装CUDA toolkit
去[官网](https://developer.nvidia.com/cuda-toolkit-archive)下载11.8的CUDA toolkit（2023.9.30，tensorflow2.14）  
安装完成后确认`/usr/local/cuda/bin/nvcc --version`，用local或者runfile方式安装！network会自动安装成最新版本  
Ubuntu似乎可以`sudo apt install nvidia-cuda-toolkit`这么安装（但我没试过是否可行）  
## 装cuDNN
去[官网](https://developer.nvidia.com/rdp/cudnn-download)下载对于11.x CUDA的v8.9.5的cuDNN（2023.9.30，tensorflow2.14）
（如果将来有报错考虑`sudo apt install zlib1g`）
  
# 装anaconda
把anaconda的linux版本的安装包下载并复制到wsl的home路径下，像Linux一样安装并添加路径：  
在`~/.bashrc·`文件末尾添加`export PATH=/home/USER_NAME/anaconda3/bin:$PATH`后`source ~/.bashrc`  


