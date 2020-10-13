# horovod-setup
This setup guide describes how to make an Ubuntu 20.04 Server based VM image with Horovod and tensorflow
to set up a Horovod cluster for distributed training.

## Install CMake

```bash
$ wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-Linux-x86_64.tar.gz
$ tar xvfz cmake-3.18.4-Linux-x86_64.tar.gz
$ echo ~/cmake-3.18.4-Linux-x86_64/bin >> ~/.bashrc
$ bash
```

## Install OpenMPI

```bash
$ sudo apt-get update -y
$ sudo apt-get install -y openmpi-bin
```

## Install venv

```bash
$ sudo apt-get install -y python3-venv
```

## Install g++

```bash
$ sudo apt install g++
```

## Install CUDA and GPU drivers

```bash
$ sudo apt install -y nvidia-cuda-toolkit
$ sudo add-apt-repository ppa:graphics-drivers
$ sudo apt update -y
$ sudo apt upgrade -y
$ sudo apt install -y nvidia-driver-440
```

Then reboot and run ``run nvidia-smi`` to verify that the driver is working.

Note that tensorflow recommends using Cuda 10.1 and cuDNN 7.6.5.

## Check CUDA version

```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
(hovorod) ubuntu@ip-172-31-92-187:~$
```

## Install cuDNN

Make a Developer Account in NVIDIA to install the cuDNN library. Then add the ``lib64`` directory to
``LD_LIBRARY_PATH``.

```bash
$ echo export LD_LIBRARY_PATH=/home/ubuntu/cudnn-10.1-linux-x64-v7.6.5.32/lib64 >> ~/.bashrc
```

## Install NCCL

Download the NCCL repo using your NVIDIA developer account.

```bash
$ sudo dpkg -i nccl-repo-ubuntu1804-2.7.8-ga-cuda10.1_1-1_amd64.deb
$ sudo apt update -y
4 sudo apt upgrade -y
$ sudo apt install libnccl2=2.7.8-1+cuda10.1 libnccl-dev=2.7.8-1+cuda10.1
```

Add the NCCL shared object file in ``LD_LIBRARY_PATH``.

```bash
$ echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu >> ~/.bashrc
```

Ubuntu 20.04 comes bundled with gcc9/g++9. Configure your session to use gcc8/g++8 instead. Horovod
doesn't support gcc9 and above.

```bash
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
```

Now we are ready to install``horovod``. Create a python3 venv called ``horovod`` and install ``tensorrflow``.

```bash
$ python -m venv horovod
$source horovod/bin/activate
(horovod)$ pip install tensorflow
(horovod)$ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITHOUT_GLOO=1 pip install --no-cache horovod
```

Verify that tensorflow can see the GPU as a visible device.

```python
import tensorflow as tf
tf.config.get_visible_devices('GPU')
```

From another node, open multiple terminals and execute ``horovodrun``:

```bash
horovodrun -np 1 aws-gpu:1 python train.py
```
