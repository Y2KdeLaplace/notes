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
添加nvcc到环境变量: 在`~/.bashrc·`文件末尾添加`export PATH=$PATH:/usr/local/cuda/bin`
## 装cuDNN
去[官网](https://developer.nvidia.com/rdp/cudnn-download)下载对于11.x CUDA的v8.9.5的cuDNN（2023.9.30，tensorflow2.14）  
安装依赖`sudo apt install zlib1g`  
安装方式参考[官方文档](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)的
>Before issuing the following commands, you must replace X.Y and v8.x.x.x with your specific CUDA and cuDNN versions and package date.  
>  
>Navigate to your <cudnnpath> directory containing the cuDNN tar file.  
>Unzip the cuDNN package.  
> `tar -xvf cudnn-linux-$arch-8.x.x.x_cudaX.Y-archive.tar.xz`  
>  
>Where `$arch` is x86_64, sbsa, or ppc64le.  
>Copy the following files into the CUDA toolkit directory.  
>`sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include` 
>`sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64`
>`sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*`
  
# 装anaconda
把anaconda的linux版本的安装包下载并复制到wsl的home路径下，像Linux一样安装并添加路径：  
在`~/.bashrc·`文件末尾添加`export PATH=/home/USER_NAME/anaconda3/bin:$PATH`，并`source ~/.bashrc`  


