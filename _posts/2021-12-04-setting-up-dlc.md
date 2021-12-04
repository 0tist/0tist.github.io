---
layout: post
title: "Setting up DeepLabCut with CUDA in Ubuntu"
comments: true
description: "Setting up cuda and cudnn with wxpython for deeplabcut in ubuntu 20.04(other versions also supported)"
---
I've struggles quite a bit trying to install the right CUDA version and some of the dependencies that are required Tensorflow to work with GPU.
As a result, I decided to make a recipe for my future self and others.
Most of the steps that I've mentioned below are from https://deeplabcut.github.io/DeepLabCut/docs/recipes/installTips.html 
This is an attempt to make it more general for varied Ubuntu versions and GPUs.
## Installing CUDA for the GPU
- `sudo apt install gcc`
- Go to the Link: https://developer.nvidia.com/cuda-downloads
	- Select `Ubuntu`
	- You can check your architecture type with command `arch` on your terminal
		- Select the architecture type(mine is `x86_64`)
	- Select your Linux Destro Type
		- Followed by Version
	- Lastly, Installer Type, Since i'm installing it locally so the install type is `deb(local)`
Doing this will provide you with a list of commands, For example: 
![](https://raw.githubusercontent.com/0tist/0tist.github.io/master/assets/images/cuda-com.png)
After following these commands you'll observe 2 cuda folders in `/usr/local/`
_Optional_ commands:
```
		sudo add-apt-repository ppa:graphics-drivers/ppa
		sudo apt update
		sudo ubuntu-drivers autoinstall
```
- And `reboot`
- Verify the installation by
```
gcc --version
nvcc --version
nvidia-smi
```
- If any of these commands return an error, you can top it up with `sudo apt install nvidia-cuda-toolkit gcc-9`
	- <ins>Caveat</ins>:
		- apt-installing nvidia-cuda-toolkit installs another cuda version on the device, as a result you might see 3 cuda folder in `/usr/local/`
## Installing Anaconda
- Go to: https://www.anaconda.com/products/individual#Downloads
- Choose your installer
	- For ubuntu, you can install by `cd`_ing_ into the folder where the installer is downloaded and run `bash <installer_file.sh>`
- Verify the install by `conda --version`

## Installing DeepLabCut
- `sudo apt install libcanberra-gtk-module libcanberra-gtk3-module`
- `git clone https://github.com/DeepLabCut/DeepLabCut.git`
- open DeepLabCut/conda-environments/**DEEPLABCUT.yaml**
- change the following:
```
	-pip:
		- "deeplabcut[gui]"
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;to:
```
	-pip:
		- deeplabcut
```
- run `conda env create -f DEEPLABCUT.yaml`
- `conda activate DEEPLABCUT`
- after you are done building the environment and installing deeplabcut within the environment, the next obstacle that we might face is GUI not working. DeepLabCut uses wxPython for its GUI, but wxPython doesn't have a general wheel for all Linux destros, so we need to install a specific wheel, which you can look up on https://extras.wxpython.org/wxPython4/extras/linux/gtk3, since I'm using Ubuntu 20.04, I'll run `pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04 wxPython`
- `conda install -c conda-forge wxpython`
- To verify the installation run `python -m deeplabcut` which should launch the DLC GUI.

## Last Obstacle
- Although you have installed CUDA but you might not be able to use GPU. Let's check it once before stating anything concretely.
- In the `ipython`console:
```python
import tensorflow as tf
tf.test.is_gpu_available()
```
`If` this returns _True_ your setup is COMPLETE! <br>
`Else`you just need to wait a lil longer and we'll surely address this
- You might be getting an error like this:
	```
	ImportError: libcudart.so.8.0: cannot open shared object file: No such file or directory
	```
- You need to install cudnn files from https://developer.nvidia.com/rdp/cudnn-download
	- You might have to create an account and participate in a survey if you haven't already done that.
	- Post downloading it, you can install the software just by double clicking on it.
			-`software install` will pop up and it will look something like this:
			![](https://raw.githubusercontent.com/0tist/0tist.github.io/master/assets/images/sft_install.png)
	- To verify the aforementioned installation look for `libcudnn.so.8.3.1` or similar named file in `/usr/lib/x86_64-linux-gnu`.
	- Now `cd` into all the `cuda*/lib64` folder in `/usr/local/`
		- For example: `/usr/local/cuda-11/lib64`
	- and enter the following command
	```
	sudo ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8.3.1 libcudnn.so.8
	sudo ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8.3.1 libcudnn.so
	```
	- You are good to go!

Thank you reading!
I hope it was useful.
