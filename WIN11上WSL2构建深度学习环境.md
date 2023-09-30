# 装子系统
1.在系统设置里面打开子系统和虚拟机设置  
2.wsl --install  
（看情况运行wsl --set-default-version 2）  
（去微软商店安装ubuntu wsl -l -v看wsl有的环境和版本）  

# 装CUDA配件
## 装CUDA驱动
`nvidia-smi`看看有没有驱动，没有就去给win11系统装驱动，有驱动的话看看cuda版本是多少，这决定之后CUDA toolkit和cudnn怎么装
## 装CUDA toolkit


# 装anaconda
把anaconda的linux版本的安装包下载并复制到wsl的home路径下，像Linux一样安装并添加路径：  
在`~/.bashrc·`文件末尾添加`export PATH=/home/USER_NAME/anaconda3/bin:$PATH`后`source ~/.bashrc`  


