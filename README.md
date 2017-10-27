# hello_world
tutorial to github


[root@cobalt cobalt]# FIREFOX
bash: FIREFOX: command not found
[root@cobalt cobalt]# ls
Android
android-studio
android-studio-ide-141.1903250-linux.zip
AndroidStudioProjects
backup
gradle-2.4
gradle-2.4-all.zip
libdeep
[root@cobalt cobalt]# cd /
[root@cobalt /]# ls
bin   etc   lib64  opt   run   sys  var
boot  home  media  proc  sbin  tmp
dev   lib   mnt    root  srv   usr
[root@cobalt /]# cd use
bash: cd: use: No such file or directory
[root@cobalt /]# ls
bin   etc   lib64  opt   run   sys  var
boot  home  media  proc  sbin  tmp
dev   lib   mnt    root  srv   usr
[root@cobalt /]# cd usr
[root@cobalt usr]# ls
bin  games    lib    libexec  sbin   src
etc  include  lib64  local    share  tmp
[root@cobalt usr]# cd src
[root@cobalt src]# ls
debug  kernels
[root@cobalt src]# cd kernels
[root@cobalt kernels]# ls
3.10.0-229.4.2.el7.x86_64
[root@cobalt kernels]# cd ..
[root@cobalt src]# ls
debug  kernels
[root@cobalt src]# cd ..
[root@cobalt usr]# ls
bin  games    lib    libexec  sbin   src
etc  include  lib64  local    share  tmp
[root@cobalt usr]# cd ..
[root@cobalt /]# ls
bin   etc   lib64  opt   run   sys  var
boot  home  media  proc  sbin  tmp
dev   lib   mnt    root  srv   usr
[root@cobalt /]# cd root
\[root@cobalt ~]# ls
anaconda         Desktop            src
anaconda-ks.cfg  Downloads          Untitled.ipynb
deep video       sports-1m-dataset
[root@cobalt ~]# cd src
[root@cobalt src]# ls
Anaconda-2.2.0-Linux-x86_64.sh
caffe
OpenBLAS
opencv
rhscl-python27-epel-7-x86_64-1-2.noarch.rpm
Theano
Untitled.ipynb
[root@cobalt src]# cd caffe
[root@cobalt caffe]# ls
build            LICENSE
caffe.cloc       Makefile
cmake            Makefile.config
CMakeLists.txt   Makefile.config.example
CMakeLists.txt~  matlab
CONTRIBUTORS.md  models
data             python
distribute       README.md
docs             scripts
examples         src
include          tools
INSTALL.md
[root@cobalt caffe]# foo =blob.cpu_data()
bash: syntax error near unexpected token `('
[root@cobalt caffe]# foo =blob.cpu_data();
bash: syntax error near unexpected token `('
[root@cobalt caffe]# ./cmdcaffe
bash: ./cmdcaffe: No such file or directory
[root@cobalt caffe]# locate cmdcaffe
[root@cobalt caffe]# cd docs
[root@cobalt docs]# ls
CMakeLists.txt   install_osx.md
CNAME            install_yum.md
_config.yml      _layouts
development.md   model_zoo.md
images           performance_hardware.md
index.md         README.md
install_apt.md   stylesheets
installation.md  tutorial
[root@cobalt docs]# cd tutorial/
[root@cobalt tutorial]# ls
convolution.md       interfaces.md
data.md              layers.md
fig                  loss.md
forward_backward.md  net_layer_blob.md
index.md             solver.md
[root@cobalt tutorial]# cd ..
[root@cobalt docs]# ls
CMakeLists.txt   install_osx.md
CNAME            install_yum.md
_config.yml      _layouts
development.md   model_zoo.md
images           performance_hardware.md
index.md         README.md
install_apt.md   stylesheets
installation.md  tutorial
[root@cobalt docs]# cat README.md 
# Caffe Documentation

To generate the documentation, run `$CAFFE_ROOT/scripts/build_docs.sh`.

To push your changes to the documentation to the gh-pages branch of your or the BVLC repo, run `$CAFFE_ROOT/scripts/deploy_docs.sh <repo_name>`.
[root@cobalt docs]# ls
CMakeLists.txt   install_osx.md
CNAME            install_yum.md
_config.yml      _layouts
development.md   model_zoo.md
images           performance_hardware.md
index.md         README.md
install_apt.md   stylesheets
installation.md  tutorial
[root@cobalt docs]# cd ..
[root@cobalt caffe]# ls
build            LICENSE
caffe.cloc       Makefile
cmake            Makefile.config
CMakeLists.txt   Makefile.config.example
CMakeLists.txt~  matlab
CONTRIBUTORS.md  models
data             python
distribute       README.md
docs             scripts
examples         src
include          tools
INSTALL.md
[root@cobalt caffe]# cd examples
[root@cobalt examples]# ls
cifar10
classification.ipynb
CMakeLists.txt
detection.ipynb
feature_extraction
filter_visualization.ipynb
finetune_flickr_style
finetune_pascal_detection
hdf5_classification
hdf5_classification.ipynb
imagenet
images
mnist
net_surgery
net_surgery.ipynb
siamese
web_demo
[root@cobalt examples]# locate *.proto.txt
[root@cobalt examples]# locate *.prototxt
/root/src/caffe/examples/cifar10/cifar10_full.prototxt
/root/src/caffe/examples/cifar10/cifar10_full_solver.prototxt
/root/src/caffe/examples/cifar10/cifar10_full_solver_lr1.prototxt
/root/src/caffe/examples/cifar10/cifar10_full_solver_lr2.prototxt
/root/src/caffe/examples/cifar10/cifar10_full_train_test.prototxt
/root/src/caffe/examples/cifar10/cifar10_quick.prototxt
/root/src/caffe/examples/cifar10/cifar10_quick_solver.prototxt
/root/src/caffe/examples/cifar10/cifar10_quick_solver_lr1.prototxt
/root/src/caffe/examples/cifar10/cifar10_quick_train_test.prototxt
/root/src/caffe/examples/feature_extraction/imagenet_val.prototxt
/root/src/caffe/examples/finetune_pascal_detection/pascal_finetune_solver.prototxt
/root/src/caffe/examples/finetune_pascal_detection/pascal_finetune_trainval_test.prototxt
/root/src/caffe/examples/hdf5_classification/solver.prototxt
/root/src/caffe/examples/hdf5_classification/solver2.prototxt
/root/src/caffe/examples/hdf5_classification/train_val.prototxt
/root/src/caffe/examples/hdf5_classification/train_val2.prototxt
/root/src/caffe/examples/mnist/lenet.prototxt
/root/src/caffe/examples/mnist/lenet_consolidated_solver.prototxt
/root/src/caffe/examples/mnist/lenet_multistep_solver.prototxt
/root/src/caffe/examples/mnist/lenet_solver.prototxt
/root/src/caffe/examples/mnist/lenet_stepearly_solver.prototxt
/root/src/caffe/examples/mnist/lenet_train_test.prototxt
/root/src/caffe/examples/mnist/mnist_autoencoder.prototxt
/root/src/caffe/examples/mnist/mnist_autoencoder_solver.prototxt
/root/src/caffe/examples/mnist/mnist_autoencoder_solver_adagrad.prototxt
/root/src/caffe/examples/mnist/mnist_autoencoder_solver_nesterov.prototxt
/root/src/caffe/examples/net_surgery/bvlc_caffenet_full_conv.prototxt
/root/src/caffe/examples/net_surgery/conv.prototxt
/root/src/caffe/examples/siamese/mnist_siamese.prototxt
/root/src/caffe/examples/siamese/mnist_siamese_solver.prototxt
/root/src/caffe/examples/siamese/mnist_siamese_train_test.prototxt
/root/src/caffe/models/bvlc_alexnet/deploy.prototxt
/root/src/caffe/models/bvlc_alexnet/solver.prototxt
/root/src/caffe/models/bvlc_alexnet/train_val.prototxt
/root/src/caffe/models/bvlc_googlenet/deploy.prototxt
/root/src/caffe/models/bvlc_googlenet/quick_solver.prototxt
/root/src/caffe/models/bvlc_googlenet/solver.prototxt
/root/src/caffe/models/bvlc_googlenet/train_val.prototxt
/root/src/caffe/models/bvlc_reference_caffenet/deploy.prototxt
/root/src/caffe/models/bvlc_reference_caffenet/solver.prototxt
/root/src/caffe/models/bvlc_reference_caffenet/train_val.prototxt
/root/src/caffe/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt
/root/src/caffe/models/finetune_flickr_style/deploy.prototxt
/root/src/caffe/models/finetune_flickr_style/solver.prototxt
/root/src/caffe/models/finetune_flickr_style/train_val.prototxt
[root@cobalt examples]# cd /root/src/caffe/examples/siamese/mnist_siamese.prototxt
bash: cd: /root/src/caffe/examples/siamese/mnist_siamese.prototxt: Not a directory
[root@cobalt examples]# cat /root/src/caffe/examples/siamese/mnist_siamese.prototxt
name: "mnist_siamese"
input: "data"
input_dim: 10000
input_dim: 1
input_dim: 28
input_dim: 28
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 500
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10
  }
}
layer {
  name: "feat"
  type: "InnerProduct"
  bottom: "ip2"
  top: "feat"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 2
  }
}
[root@cobalt examples]# locate *.prototxt
/root/src/caffe/examples/cifar10/cifar10_full.prototxt
/root/src/caffe/examples/cifar10/cifar10_full_solver.prototxt
/root/src/caffe/examples/cifar10/cifar10_full_solver_lr1.prototxt
/root/src/caffe/examples/cifar10/cifar10_full_solver_lr2.prototxt
/root/src/caffe/examples/cifar10/cifar10_full_train_test.prototxt
/root/src/caffe/examples/cifar10/cifar10_quick.prototxt
/root/src/caffe/examples/cifar10/cifar10_quick_solver.prototxt
/root/src/caffe/examples/cifar10/cifar10_quick_solver_lr1.prototxt
/root/src/caffe/examples/cifar10/cifar10_quick_train_test.prototxt
/root/src/caffe/examples/feature_extraction/imagenet_val.prototxt
/root/src/caffe/examples/finetune_pascal_detection/pascal_finetune_solver.prototxt
/root/src/caffe/examples/finetune_pascal_detection/pascal_finetune_trainval_test.prototxt
/root/src/caffe/examples/hdf5_classification/solver.prototxt
/root/src/caffe/examples/hdf5_classification/solver2.prototxt
/root/src/caffe/examples/hdf5_classification/train_val.prototxt
/root/src/caffe/examples/hdf5_classification/train_val2.prototxt
/root/src/caffe/examples/mnist/lenet.prototxt
/root/src/caffe/examples/mnist/lenet_consolidated_solver.prototxt
/root/src/caffe/examples/mnist/lenet_multistep_solver.prototxt
/root/src/caffe/examples/mnist/lenet_solver.prototxt
/root/src/caffe/examples/mnist/lenet_stepearly_solver.prototxt
/root/src/caffe/examples/mnist/lenet_train_test.prototxt
/root/src/caffe/examples/mnist/mnist_autoencoder.prototxt
/root/src/caffe/examples/mnist/mnist_autoencoder_solver.prototxt
/root/src/caffe/examples/mnist/mnist_autoencoder_solver_adagrad.prototxt
/root/src/caffe/examples/mnist/mnist_autoencoder_solver_nesterov.prototxt
/root/src/caffe/examples/net_surgery/bvlc_caffenet_full_conv.prototxt
/root/src/caffe/examples/net_surgery/conv.prototxt
/root/src/caffe/examples/siamese/mnist_siamese.prototxt
/root/src/caffe/examples/siamese/mnist_siamese_solver.prototxt
/root/src/caffe/examples/siamese/mnist_siamese_train_test.prototxt
/root/src/caffe/models/bvlc_alexnet/deploy.prototxt
/root/src/caffe/models/bvlc_alexnet/solver.prototxt
/root/src/caffe/models/bvlc_alexnet/train_val.prototxt
/root/src/caffe/models/bvlc_googlenet/deploy.prototxt
/root/src/caffe/models/bvlc_googlenet/quick_solver.prototxt
/root/src/caffe/models/bvlc_googlenet/solver.prototxt
/root/src/caffe/models/bvlc_googlenet/train_val.prototxt
/root/src/caffe/models/bvlc_reference_caffenet/deploy.prototxt
/root/src/caffe/models/bvlc_reference_caffenet/solver.prototxt
/root/src/caffe/models/bvlc_reference_caffenet/train_val.prototxt
/root/src/caffe/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt
/root/src/caffe/models/finetune_flickr_style/deploy.prototxt
/root/src/caffe/models/finetune_flickr_style/solver.prototxt
/root/src/caffe/models/finetune_flickr_style/train_val.prototxt
[root@cobalt examples]# 
[root@cobalt examples]# 
[root@cobalt examples]# PWD
bash: PWD: command not found
[root@cobalt examples]# pwd
/root/src/caffe/examples
[root@cobalt examples]# 
[root@cobalt examples]# pwd
/root/src/caffe/examples
[root@cobalt examples]# cd ..
\[root@cobalt caffe]# \cd ..
[root@cobalt src]# ls
Anaconda-2.2.0-Linux-x86_64.sh  OpenBLAS  rhscl-python27-epel-7-x86_64-1-2.noarch.rpm  Untitled.ipynb
caffe                           opencv    Theano
[root@cobalt src]# cd ..
[root@cobalt ~]# ls
anaconda  anaconda-ks.cfg  deep video  Desktop  Downloads  sports-1m-dataset  src  Untitled.ipynb
[root@cobalt ~]# cd ..
[root@cobalt /]# ls
bin  boot  dev  etc  home  lib  lib64  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var
[root@cobalt /]# cd root
[root@cobalt ~]# ls
anaconda  anaconda-ks.cfg  deep video  Desktop  Downloads  sports-1m-dataset  src  Untitled.ipynb
[root@cobalt ~]# ls
anaconda  anaconda-ks.cfg  deep video  Desktop  Downloads  sports-1m-dataset  src  Untitled.ipynb
[root@cobalt ~]# cd anaconda
[root@cobalt anaconda]# ls
bin         envs  Examples  include  LICENSE.txt  pkgs     share  var
conda-meta  etc   imports   lib      mkspecs      plugins  ssl
[root@cobalt anaconda]# cd ..
[root@cobalt ~]# ls
anaconda  anaconda-ks.cfg  deep video  Desktop  Downloads  sports-1m-dataset  src  Untitled.ipynb
[root@cobalt ~]# cd src
[root@cobalt src]# ls
Anaconda-2.2.0-Linux-x86_64.sh  OpenBLAS  rhscl-python27-epel-7-x86_64-1-2.noarch.rpm  Untitled.ipynb
caffe                           opencv    Theano
[root@cobalt src]# cd caffe
[root@cobalt caffe]# ls
build           CMakeLists.txt~  docs        LICENSE                  matlab     scripts
caffe.cloc      CONTRIBUTORS.md  examples    Makefile                 models     src
cmake           data             include     Makefile.config          python     tools
CMakeLists.txt  distribute       INSTALL.md  Makefile.config.example  README.md
[root@cobalt caffe]# cat Makefile.config
## Refer to http://caffe.berkeleyvision.org/installation.html
# Contributions simplifying and improving our build system are welcome!

# cuDNN acceleration switch (uncomment to build with cuDNN).
# USE_CUDNN := 1

# CPU-only switch (uncomment to build without GPU support).
 CPU_ONLY := 1

# To customize your choice of compiler, uncomment and set the following.
# N.B. the default for Linux is g++ and the default for OSX is clang++
# CUSTOM_CXX := g++

# CUDA directory contains bin/ and lib/ directories that we need.
CUDA_DIR := /usr/local/cuda
# On Ubuntu 14.04, if cuda tools are installed via
# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
# CUDA_DIR := /usr

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 lines for compatibility.
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
		-gencode arch=compute_20,code=sm_21 \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_50,code=compute_50

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
BLAS := open
# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
 BLAS_INCLUDE := /opt/openblas/include
 BLAS_LIB := /opt/openblas/lib

# This is required only if you will compile the matlab interface.
# MATLAB directory should contain the mex binary in /bin.
# MATLAB_DIR := /usr/local
# MATLAB_DIR := /Applications/MATLAB_R2012b.app

# NOTE: this is required only if you will compile the python interface.
# We need to be able to find Python.h and numpy/arrayobject.h.
# PYTHON_INCLUDE := /usr/include/python2.7 \
#		/usr/lib/python2.7/dist-packages/numpy/core/include
# Anaconda Python distribution is quite popular. Include path:
# Verify anaconda location, sometimes it's in root.
 ANACONDA_HOME := /root/anaconda
 PYTHON_INCLUDE := /root/anaconda/include \
                   /root/anaconda/include/python2.7 \
                   /root/anaconda/lib/python2.7/site-packages/numpy/core/include \

# We need to be able to find libpythonX.X.so or .dylib.
# PYTHON_LIB := /usr/lib
  PYTHON_LIB := /root/anaconda/lib

# Uncomment to support layers written in Python (will link against Python libs)
  WITH_PYTHON_LAYER := 1

# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib

# Uncomment to use `pkg-config` to specify OpenCV library paths.
# (Usually not necessary -- OpenCV libraries are normally installed in one of the above $LIBRARY_DIRS.)
 USE_PKG_CONFIG := 1

BUILD_DIR := build
DISTRIBUTE_DIR := distribute

# Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
# DEBUG := 1

# The ID of the GPU that 'make runtest' will use to run unit tests.
TEST_GPUID := 0

# enable pretty build (comment to see full commands)
Q ?= @[root@cobalt caffe]# ./data/mnist/get_mnist.sh
Downloading...
--2015-06-16 21:08:21--  http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Resolving proxy.dlsu.edu.ph (proxy.dlsu.edu.ph)... 192.168.13.11, 192.168.13.10
Connecting to proxy.dlsu.edu.ph (proxy.dlsu.edu.ph)|192.168.13.11|:80... connected.
Proxy request sent, awaiting response... 200 OK
Length: 9912422 (9.5M) [application/x-gzip]
Saving to: ‘train-images-idx3-ubyte.gz’

100%[==============================================================>] 9,912,422   59.0KB/s   in 3m 49s 

2015-06-16 21:12:11 (42.4 KB/s) - ‘train-images-idx3-ubyte.gz’ saved [9912422/9912422]

--2015-06-16 21:12:11--  http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Resolving proxy.dlsu.edu.ph (proxy.dlsu.edu.ph)... 192.168.13.10, 192.168.13.11
Connecting to proxy.dlsu.edu.ph (proxy.dlsu.edu.ph)|192.168.13.10|:80... connected.
Proxy request sent, awaiting response... 200 OK
Length: 28881 (28K) [application/x-gzip]
Saving to: ‘train-labels-idx1-ubyte.gz’

100%[==============================================================>] 28,881      --.-K/s   in 0.004s  

2015-06-16 21:12:12 (7.02 MB/s) - ‘train-labels-idx1-ubyte.gz’ saved [28881/28881]

--2015-06-16 21:12:12--  http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Resolving proxy.dlsu.edu.ph (proxy.dlsu.edu.ph)... 192.168.13.10, 192.168.13.11
Connecting to proxy.dlsu.edu.ph (proxy.dlsu.edu.ph)|192.168.13.10|:80... connected.
Proxy request sent, awaiting response... 200 OK
Length: 1648877 (1.6M) [application/x-gzip]
Saving to: ‘t10k-images-idx3-ubyte.gz’

100%[==============================================================>] 1,648,877   --.-K/s   in 0.1s    

2015-06-16 21:12:12 (10.8 MB/s) - ‘t10k-images-idx3-ubyte.gz’ saved [1648877/1648877]

--2015-06-16 21:12:12--  http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Resolving proxy.dlsu.edu.ph (proxy.dlsu.edu.ph)... 192.168.13.11, 192.168.13.10
Connecting to proxy.dlsu.edu.ph (proxy.dlsu.edu.ph)|192.168.13.11|:80... connected.
Proxy request sent, awaiting response... 200 OK
Length: 4542 (4.4K) [application/x-gzip]
Saving to: ‘t10k-labels-idx1-ubyte.gz’

100%[==============================================================>] 4,542       --.-K/s   in 0.001s  

2015-06-16 21:12:13 (8.50 MB/s) - ‘t10k-labels-idx1-ubyte.gz’ saved [4542/4542]

Unzipping...
Done.
[root@cobalt caffe]# ./examples/mnist/create_mnist.sh
Creating lmdb...
build/examples/mnist/convert_mnist_data.bin: error while loading shared libraries: libhdf5_hl.so.9: cannot open shared object file: No such file or directory
build/examples/mnist/convert_mnist_data.bin: error while loading shared libraries: libhdf5_hl.so.9: cannot open shared object file: No such file or directory
Done.
[root@cobalt caffe]# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/anaconda/lib"
[root@cobalt caffe]# ./examples/mnist/create_mnist.sh
Creating lmdb...
Done.
[root@cobalt caffe]#  echo $PATH
/home/cobalt/Android/Sdk/platform-tools:/root/anaconda/bin:/home/cobalt/Android/Sdk/platform-tools:/root/anaconda/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin:/opt/jdk1.7.0_79/bin:/opt/jdk1.7.0_79/jre/bin:/root/bin:/opt/jdk1.7.0_79/bin:/opt/jdk1.7.0_79/jre/bin
[root@cobalt caffe]# cd examples/mnist/
You have new mail in /var/spool/mail/root
[root@cobalt mnist]# ls
convert_mnist_data.cpp                     mnist_autoencoder_solver_nesterov.prototxt
create_mnist.sh                            mnist_autoencoder_solver.prototxt
lenet_consolidated_solver.prototxt         mnist_test_lmdb
lenet_multistep_solver.prototxt            mnist_train_lmdb
lenet.prototxt                             readme.md
lenet_solver.prototxt                      train_lenet_consolidated.sh
lenet_stepearly_solver.prototxt            train_lenet.sh
lenet_train_test.prototxt                  train_mnist_autoencoder_adagrad.sh
mnist_autoencoder.prototxt                 train_mnist_autoencoder_nesterov.sh
mnist_autoencoder_solver_adagrad.prototxt  train_mnist_autoencoder.sh
[root@cobalt mnist]# python
Python 2.7.9 |Anaconda 2.2.0 (64-bit)| (default, Apr 14 2015, 12:54:25) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
Anaconda is brought to you by Continuum Analytics.
Please check out: http://continuum.io/thanks and https://binstar.org
>>> name "LeNet"
  File "<stdin>", line 1
    name "LeNet"
               ^
SyntaxError: invalid syntax
>>> name: "LeNet"
  File "<stdin>", line 1
    name: "LeNet"
        ^
SyntaxError: invalid syntax
>>> 
>>> 
>>> 
>>> exit
Use exit() or Ctrl-D (i.e. EOF) to exit
>>> quit
Use quit() or Ctrl-D (i.e. EOF) to exit
>>> 
[root@cobalt mnist]# cd ..
[root@cobalt examples]# ls
cifar10               filter_visualization.ipynb  imagenet           siamese
classification.ipynb  finetune_flickr_style       images             web_demo
CMakeLists.txt        finetune_pascal_detection   mnist
detection.ipynb       hdf5_classification         net_surgery
feature_extraction    hdf5_classification.ipynb   net_surgery.ipynb
[root@cobalt examples]# cd mnist
[root@cobalt mnist]# cd ..
[root@cobalt examples]# cd ..
[root@cobalt caffe]# git pull
remote: Counting objects: 912, done.
remote: Compressing objects: 100% (8/8), done.
remote: Total 912 (delta 526), reused 525 (delta 525), pack-reused 379
Receiving objects: 100% (912/912), 1.21 MiB | 56.00 KiB/s, done.
Resolving deltas: 100% (615/615), completed with 148 local objects.
From https://github.com/BVLC/caffe
   c255709..0d7c6cb  master     -> origin/master
   20c7449..b7c55d7  gh-pages   -> origin/gh-pages
Updating c255709..0d7c6cb
error: Your local changes to the following files would be overwritten by merge:
	CMakeLists.txt
	cmake/Dependencies.cmake
Please, commit your changes or stash them before you can merge.
Aborting
[root@cobalt caffe]# ls
build           CMakeLists.txt~  docs        LICENSE                  matlab     scripts
caffe.cloc      CONTRIBUTORS.md  examples    Makefile                 models     src
cmake           data             include     Makefile.config          python     tools
CMakeLists.txt  distribute       INSTALL.md  Makefile.config.example  README.md
[root@cobalt caffe]# mv CMakeLists.txt CMakeLists_old.txt
[root@cobalt caffe]# mv cmake/Dependencies.cmake cmake/Dependencies_old.cmake
[root@cobalt caffe]# ls
build               CMakeLists.txt~  docs        LICENSE                  matlab     scripts
caffe.cloc          CONTRIBUTORS.md  examples    Makefile                 models     src
cmake               data             include     Makefile.config          python     tools
CMakeLists_old.txt  distribute       INSTALL.md  Makefile.config.example  README.md
[root@cobalt caffe]# git pull
Updating c255709..0d7c6cb
Fast-forward
 CMakeLists.txt                                          |   2 +
 Makefile                                                |  16 +-
 Makefile.config.example                                 |  12 +
 cmake/ConfigGen.cmake                                   |  11 +
 cmake/Dependencies.cmake                                |   4 +-
 cmake/External/gflags.cmake                             |  56 +++++
 cmake/External/glog.cmake                               |  56 +++++
 cmake/Targets.cmake                                     |   4 +
 docs/install_osx.md                                     |  12 +-
 docs/tutorial/interfaces.md                             | 211 +++++++++++++++-
 docs/tutorial/layers.md                                 |   6 +-
 examples/cpp_classification/classification.cpp          | 255 +++++++++++++++++++
 examples/cpp_classification/readme.md                   |  77 ++++++
 examples/imagenet/readme.md                             |   4 +-
 examples/mnist/readme.md                                |   2 +-
 examples/net_surgery.ipynb                              |  88 ++++++-
 examples/web_demo/app.py                                |   2 +-
 include/caffe/common_layers.hpp                         | 108 ++++++++
 include/caffe/data_transformer.hpp                      |  36 +++
 include/caffe/neuron_layers.hpp                         |  66 +++++
 include/caffe/solver.hpp                                |  15 +-
 include/caffe/test/test_caffe_main.hpp                  |  28 ++-
 include/caffe/test/test_gradient_check_util.hpp         |   7 +-
 include/caffe/util/db.hpp                               | 136 ----------
 include/caffe/util/db_leveldb.hpp                       |  73 ++++++
 include/caffe/util/db_lmdb.hpp                          |  91 +++++++
 include/caffe/util/math_functions.hpp                   |   6 +
 include/caffe/util/mkl_alternate.hpp                    |   1 +
 matlab/+caffe/+test/test_net.m                          |  96 +++++++
 matlab/+caffe/+test/test_solver.m                       |  45 ++++
 matlab/+caffe/Blob.m                                    |  78 ++++++
 matlab/+caffe/Layer.m                                   |  32 +++
 matlab/+caffe/Net.m                                     | 133 ++++++++++
 matlab/+caffe/Solver.m                                  |  56 +++++
 matlab/+caffe/get_net.m                                 |  37 +++
 matlab/+caffe/get_solver.m                              |  10 +
 matlab/+caffe/imagenet/ilsvrc_2012_mean.mat             | Bin 0 -> 606799 bytes
 matlab/+caffe/io.m                                      |  33 +++
 matlab/+caffe/private/CHECK.m                           |   7 +
 matlab/+caffe/private/CHECK_FILE_EXIST.m                |   7 +
 matlab/+caffe/private/caffe_.cpp                        | 546 ++++++++++++++++++++++++++++++++++++++++
 matlab/+caffe/private/is_valid_handle.m                 |  27 ++
 matlab/+caffe/reset_all.m                               |   8 +
 matlab/+caffe/run_tests.m                               |  19 ++
 matlab/+caffe/set_device.m                              |  11 +
 matlab/+caffe/set_mode_cpu.m                            |   7 +
 matlab/+caffe/set_mode_gpu.m                            |   7 +
 matlab/CMakeLists.txt                                   |   4 +-
 matlab/caffe/ilsvrc_2012_mean.mat                       | Bin 786640 -> 0 bytes
 matlab/caffe/matcaffe.cpp                               | 421 -------------------------------
 matlab/caffe/matcaffe_batch.m                           |  75 ------
 matlab/caffe/matcaffe_demo.m                            | 110 --------
 matlab/caffe/matcaffe_demo_vgg.m                        |  96 -------
 matlab/caffe/matcaffe_demo_vgg_mean_pix.m               | 102 --------
 matlab/caffe/matcaffe_init.m                            |  41 ---
 matlab/caffe/prepare_batch.m                            |  41 ---
 matlab/caffe/print_cell.m                               |  42 ----
 matlab/caffe/read_cell.m                                |  21 --
 matlab/demo/classification_demo.m                       | 147 +++++++++++
 matlab/{caffe => }/hdf5creation/.gitignore              |   0
 matlab/{caffe => }/hdf5creation/demo.m                  |   0
 matlab/{caffe => }/hdf5creation/store2hdf5.m            |   0
 models/bvlc_googlenet/train_val.prototxt                |  63 -----
 python/caffe/draw.py                                    | 163 +++++++-----
 python/draw_net.py                                      |  13 +-
 scripts/travis/travis_install.sh                        |   8 +-
 src/caffe/data_transformer.cpp                          | 116 ++++++++-
 src/caffe/layers/base_data_layer.cpp                    |  10 +-
 src/caffe/layers/base_data_layer.cu                     |   6 +-
 src/caffe/layers/concat_layer.cu                        |  44 +++-
 src/caffe/layers/conv_layer.cpp                         |   7 -
 src/caffe/layers/conv_layer.cu                          |   7 -
 src/caffe/layers/cudnn_conv_layer.cu                    |   2 -
 src/caffe/layers/data_layer.cpp                         |  90 ++-----
 src/caffe/layers/deconv_layer.cpp                       |   7 -
 src/caffe/layers/deconv_layer.cu                        |   7 -
 src/caffe/layers/filter_layer.cpp                       | 127 ++++++++++
 src/caffe/layers/filter_layer.cu                        |  70 ++++++
 src/caffe/layers/flatten_layer.cpp                      |  16 +-
 src/caffe/layers/image_data_layer.cpp                   |  42 ++--
 src/caffe/layers/inner_product_layer.cpp                |   4 +-
 src/caffe/layers/inner_product_layer.cu                 |   4 +-
 src/caffe/layers/log_layer.cpp                          |  87 +++++++
 src/caffe/layers/log_layer.cu                           |  57 +++++
 src/caffe/layers/lrn_layer.cu                           | 102 ++++----
 src/caffe/layers/pooling_layer.cu                       | 218 ++++++++--------
 src/caffe/layers/prelu_layer.cpp                        |   1 -
 src/caffe/layers/prelu_layer.cu                         |   6 +-
 src/caffe/layers/reduction_layer.cpp                    | 132 ++++++++++
 src/caffe/layers/reduction_layer.cu                     |  93 +++++++
 src/caffe/layers/slice_layer.cu                         |  47 ++--
 src/caffe/proto/caffe.proto                             |  67 ++++-
 src/caffe/solver.cpp                                    | 541 ++++++++++++++++++---------------------
 src/caffe/test/test_accuracy_layer.cpp                  |   5 +-
 src/caffe/test/test_argmax_layer.cpp                    |   3 +-
 src/caffe/test/test_convolution_layer.cpp               |   9 +-
 src/caffe/test/test_dummy_data_layer.cpp                |   5 +-
 src/caffe/test/test_filter_layer.cpp                    | 128 ++++++++++
 src/caffe/test/test_flatten_layer.cpp                   |  46 +++-
 src/caffe/test/test_gradient_based_solver.cpp           |  82 +++++-
 src/caffe/test/test_im2col_kernel.cu                    |   4 +-
 src/caffe/test/test_math_functions.cpp                  |  51 ++--
 src/caffe/test/test_multinomial_logistic_loss_layer.cpp |   3 +-
 src/caffe/test/test_neuron_layer.cpp                    | 135 +++++++++-
 src/caffe/test/test_pooling_layer.cpp                   |  13 +-
 src/caffe/test/test_reduction_layer.cpp                 | 297 ++++++++++++++++++++++
 src/caffe/test/test_softmax_layer.cpp                   |   4 +-
 src/caffe/test/test_stochastic_pooling.cpp              |  35 ++-
 src/caffe/util/db.cpp                                   |  57 +----
 src/caffe/util/db_leveldb.cpp                           |  21 ++
 src/caffe/util/db_lmdb.cpp                              |  51 ++++
 src/caffe/util/math_functions.cpp                       |  10 +
 src/caffe/util/math_functions.cu                        |  21 ++
 tools/extra/parse_log.py                                | 165 +++++++-----
 tools/extract_features.cpp                              |   5 +-
 115 files changed, 4860 insertions(+), 2091 deletions(-)
 create mode 100644 cmake/External/gflags.cmake
 create mode 100644 cmake/External/glog.cmake
 create mode 100644 examples/cpp_classification/classification.cpp
 create mode 100644 examples/cpp_classification/readme.md
 create mode 100644 include/caffe/util/db_leveldb.hpp
 create mode 100644 include/caffe/util/db_lmdb.hpp
 create mode 100644 matlab/+caffe/+test/test_net.m
 create mode 100644 matlab/+caffe/+test/test_solver.m
 create mode 100644 matlab/+caffe/Blob.m
 create mode 100644 matlab/+caffe/Layer.m
 create mode 100644 matlab/+caffe/Net.m
 create mode 100644 matlab/+caffe/Solver.m
 create mode 100644 matlab/+caffe/get_net.m
 create mode 100644 matlab/+caffe/get_solver.m
 create mode 100644 matlab/+caffe/imagenet/ilsvrc_2012_mean.mat
 create mode 100644 matlab/+caffe/io.m
 create mode 100644 matlab/+caffe/private/CHECK.m
 create mode 100644 matlab/+caffe/private/CHECK_FILE_EXIST.m
 create mode 100644 matlab/+caffe/private/caffe_.cpp
 create mode 100644 matlab/+caffe/private/is_valid_handle.m
 create mode 100644 matlab/+caffe/reset_all.m
 create mode 100644 matlab/+caffe/run_tests.m
 create mode 100644 matlab/+caffe/set_device.m
 create mode 100644 matlab/+caffe/set_mode_cpu.m
 create mode 100644 matlab/+caffe/set_mode_gpu.m
 delete mode 100644 matlab/caffe/ilsvrc_2012_mean.mat
 delete mode 100644 matlab/caffe/matcaffe.cpp
 delete mode 100644 matlab/caffe/matcaffe_batch.m
 delete mode 100644 matlab/caffe/matcaffe_demo.m
 delete mode 100644 matlab/caffe/matcaffe_demo_vgg.m
 delete mode 100644 matlab/caffe/matcaffe_demo_vgg_mean_pix.m
 delete mode 100644 matlab/caffe/matcaffe_init.m
 delete mode 100644 matlab/caffe/prepare_batch.m
 delete mode 100644 matlab/caffe/print_cell.m
 delete mode 100644 matlab/caffe/read_cell.m
 create mode 100644 matlab/demo/classification_demo.m
 rename matlab/{caffe => }/hdf5creation/.gitignore (100%)
 rename matlab/{caffe => }/hdf5creation/demo.m (100%)
 rename matlab/{caffe => }/hdf5creation/store2hdf5.m (100%)
 create mode 100644 src/caffe/layers/filter_layer.cpp
 create mode 100644 src/caffe/layers/filter_layer.cu
 create mode 100644 src/caffe/layers/log_layer.cpp
 create mode 100644 src/caffe/layers/log_layer.cu
 create mode 100644 src/caffe/layers/reduction_layer.cpp
 create mode 100644 src/caffe/layers/reduction_layer.cu
 create mode 100644 src/caffe/test/test_filter_layer.cpp
 create mode 100644 src/caffe/test/test_reduction_layer.cpp
 create mode 100644 src/caffe/util/db_leveldb.cpp
 create mode 100644 src/caffe/util/db_lmdb.cpp
[root@cobalt caffe]# ls
build               CMakeLists.txt   distribute  INSTALL.md       Makefile.config.example  README.md
caffe.cloc          CMakeLists.txt~  docs        LICENSE          matlab                   scripts
cmake               CONTRIBUTORS.md  examples    Makefile         models                   src
CMakeLists_old.txt  data             include     Makefile.config  python                   tools
[root@cobalt caffe]# cd examples/
[root@cobalt examples]# ls
cifar10               feature_extraction          hdf5_classification.ipynb  net_surgery.ipynb
classification.ipynb  filter_visualization.ipynb  imagenet                   siamese
CMakeLists.txt        finetune_flickr_style       images                     web_demo
cpp_classification    finetune_pascal_detection   mnist
detection.ipynb       hdf5_classification         net_surgery
[root@cobalt examples]# cd mnist/
[root@cobalt mnist]# ls
convert_mnist_data.cpp                     mnist_autoencoder_solver_nesterov.prototxt
create_mnist.sh                            mnist_autoencoder_solver.prototxt
lenet_consolidated_solver.prototxt         mnist_test_lmdb
lenet_multistep_solver.prototxt            mnist_train_lmdb
lenet.prototxt                             readme.md
lenet_solver.prototxt                      train_lenet_consolidated.sh
lenet_stepearly_solver.prototxt            train_lenet.sh
lenet_train_test.prototxt                  train_mnist_autoencoder_adagrad.sh
mnist_autoencoder.prototxt                 train_mnist_autoencoder_nesterov.sh
mnist_autoencoder_solver_adagrad.prototxt  train_mnist_autoencoder.sh
[root@cobalt mnist]# cd ..
[root@cobalt examples]# ls
cifar10               feature_extraction          hdf5_classification.ipynb  net_surgery.ipynb
classification.ipynb  filter_visualization.ipynb  imagenet                   siamese
CMakeLists.txt        finetune_flickr_style       images                     web_demo
cpp_classification    finetune_pascal_detection   mnist
detection.ipynb       hdf5_classification         net_surgery
[root@cobalt examples]# cd ..
[root@cobalt caffe]# ls
build               CMakeLists.txt   distribute  INSTALL.md       Makefile.config.example  README.md
caffe.cloc          CMakeLists.txt~  docs        LICENSE          matlab                   scripts
cmake               CONTRIBUTORS.md  examples    Makefile         models                   src
CMakeLists_old.txt  data             include     Makefile.config  python                   tools
[root@cobalt caffe]# cat /src/caffe/proto/caffe.proto
cat: /src/caffe/proto/caffe.proto: No such file or directory
[root@cobalt caffe]# ls
build               CMakeLists.txt   distribute  INSTALL.md       Makefile.config.example  README.md
caffe.cloc          CMakeLists.txt~  docs        LICENSE          matlab                   scripts
cmake               CONTRIBUTORS.md  examples    Makefile         models                   src
CMakeLists_old.txt  data             include     Makefile.config  python                   tools
[root@cobalt caffe]# cd src
[root@cobalt src]# ls
caffe  gtest
[root@cobalt src]# cd caffe/
[root@cobalt caffe]# ls
blob.cpp        common.cpp            internal_thread.cpp  layers   proto       syncedmem.cpp  util
CMakeLists.txt  data_transformer.cpp  layer_factory.cpp    net.cpp  solver.cpp  test
[root@cobalt caffe]# cd proto/
[root@cobalt proto]# ls
\caffe.proto
[root@cobalt proto]# \cd prot
bash: cd: prot: No such file or directory
[root@cobalt proto]# ls
caffe.proto
[root@cobalt proto]# cd pro
bash: cd: pro: No such file or directory
[root@cobalt proto]# ls
caffe.proto
[root@cobalt proto]# cat caffe.proto 
syntax = "proto2";

package caffe;

// Specifies the shape (dimensions) of a Blob.
message BlobShape {
  repeated int64 dim = 1 [packed = true];
}

message BlobProto {
  optional BlobShape shape = 7;
  repeated float data = 5 [packed = true];
  repeated float diff = 6 [packed = true];

  // 4D dimensions -- deprecated.  Use "shape" instead.
  optional int32 num = 1 [default = 0];
  optional int32 channels = 2 [default = 0];
  optional int32 height = 3 [default = 0];
  optional int32 width = 4 [default = 0];
}

// The BlobProtoVector is simply a way to pass multiple blobproto instances
// around.
message BlobProtoVector {
  repeated BlobProto blobs = 1;
}

message Datum {
  optional int32 channels = 1;
  optional int32 height = 2;
  optional int32 width = 3;
  // the actual image data, in bytes
  optional bytes data = 4;
  optional int32 label = 5;
  // Optionally, the datum could also hold float data.
  repeated float float_data = 6;
  // If true data contains an encoded image that need to be decoded
  optional bool encoded = 7 [default = false];
}

message FillerParameter {
  // The filler type.
  optional string type = 1 [default = 'constant'];
  optional float value = 2 [default = 0]; // the value in constant filler
  optional float min = 3 [default = 0]; // the min value in uniform filler
  optional float max = 4 [default = 1]; // the max value in uniform filler
  optional float mean = 5 [default = 0]; // the mean value in Gaussian filler
  optional float std = 6 [default = 1]; // the std value in Gaussian filler
  // The expected number of non-zero output weights for a given input in
  // Gaussian filler -- the default -1 means don't perform sparsification.
  optional int32 sparse = 7 [default = -1];
  // Normalize the filler variance by fan_in, fan_out, or their average.
  // Applies to 'xavier' and 'msra' fillers.
  enum VarianceNorm {
    FAN_IN = 0;
    FAN_OUT = 1;
    AVERAGE = 2;
  }
  optional VarianceNorm variance_norm = 8 [default = FAN_IN];
}

message NetParameter {
  optional string name = 1; // consider giving the network a name
  // The input blobs to the network.
  repeated string input = 3;
  // The shape of the input blobs.
  repeated BlobShape input_shape = 8;

  // 4D input dimensions -- deprecated.  Use "shape" instead.
  // If specified, for each input blob there should be four
  // values specifying the num, channels, height and width of the input blob.
  // Thus, there should be a total of (4 * #input) numbers.
  repeated int32 input_dim = 4;

  // Whether the network will force every layer to carry out backward operation.
  // If set False, then whether to carry out backward is determined
  // automatically according to the net structure and learning rates.
  optional bool force_backward = 5 [default = false];
  // The current "state" of the network, including the phase, level, and stage.
  // Some layers may be included/excluded depending on this state and the states
  // specified in the layers' include and exclude fields.
  optional NetState state = 6;

  // Print debugging information about results while running Net::Forward,
  // Net::Backward, and Net::Update.
  optional bool debug_info = 7 [default = false];

  // The layers that make up the net.  Each of their configurations, including
  // connectivity and behavior, is specified as a LayerParameter.
  repeated LayerParameter layer = 100;  // ID 100 so layers are printed last.

  // DEPRECATED: use 'layer' instead.
  repeated V1LayerParameter layers = 2;
}

// NOTE
// Update the next available ID when you add a new SolverParameter field.
//
// SolverParameter next available ID: 37 (last added: iter_size)
message SolverParameter {
  //////////////////////////////////////////////////////////////////////////////
  // Specifying the train and test networks
  //
  // Exactly one train net must be specified using one of the following fields:
  //     train_net_param, train_net, net_param, net
  // One or more test nets may be specified using any of the following fields:
  //     test_net_param, test_net, net_param, net
  // If more than one test net field is specified (e.g., both net and
  // test_net are specified), they will be evaluated in the field order given
  // above: (1) test_net_param, (2) test_net, (3) net_param/net.
  // A test_iter must be specified for each test_net.
  // A test_level and/or a test_stage may also be specified for each test_net.
  //////////////////////////////////////////////////////////////////////////////

  // Proto filename for the train net, possibly combined with one or more
  // test nets.
  optional string net = 24;
  // Inline train net param, possibly combined with one or more test nets.
  optional NetParameter net_param = 25;

  optional string train_net = 1; // Proto filename for the train net.
  repeated string test_net = 2; // Proto filenames for the test nets.
  optional NetParameter train_net_param = 21; // Inline train net params.
  repeated NetParameter test_net_param = 22; // Inline test net params.

  // The states for the train/test nets. Must be unspecified or
  // specified once per net.
  //
  // By default, all states will have solver = true;
  // train_state will have phase = TRAIN,
  // and all test_state's will have phase = TEST.
  // Other defaults are set according to the NetState defaults.
  optional NetState train_state = 26;
  repeated NetState test_state = 27;

  // The number of iterations for each test net.
  repeated int32 test_iter = 3;

  // The number of iterations between two testing phases.
  optional int32 test_interval = 4 [default = 0];
  optional bool test_compute_loss = 19 [default = false];
  // If true, run an initial test pass before the first iteration,
  // ensuring memory availability and printing the starting value of the loss.
  optional bool test_initialization = 32 [default = true];
  optional float base_lr = 5; // The base learning rate
  // the number of iterations between displaying info. If display = 0, no info
  // will be displayed.
  optional int32 display = 6;
  // Display the loss averaged over the last average_loss iterations
  optional int32 average_loss = 33 [default = 1];
  optional int32 max_iter = 7; // the maximum number of iterations
  // accumulate gradients over `iter_size` x `batch_size` instances
  optional int32 iter_size = 36 [default = 1];
  optional string lr_policy = 8; // The learning rate decay policy.
  optional float gamma = 9; // The parameter to compute the learning rate.
  optional float power = 10; // The parameter to compute the learning rate.
  optional float momentum = 11; // The momentum value.
  optional float weight_decay = 12; // The weight decay.
  // regularization types supported: L1 and L2
  // controlled by weight_decay
  optional string regularization_type = 29 [default = "L2"];
  // the stepsize for learning rate policy "step"
  optional int32 stepsize = 13;
  // the stepsize for learning rate policy "multistep"
  repeated int32 stepvalue = 34;

  // Set clip_gradients to >= 0 to clip parameter gradients to that L2 norm,
  // whenever their actual L2 norm is larger.
  optional float clip_gradients = 35 [default = -1];

  optional int32 snapshot = 14 [default = 0]; // The snapshot interval
  optional string snapshot_prefix = 15; // The prefix for the snapshot.
  // whether to snapshot diff in the results or not. Snapshotting diff will help
  // debugging but the final protocol buffer size will be much larger.
  optional bool snapshot_diff = 16 [default = false];
  // the mode solver will use: 0 for CPU and 1 for GPU. Use GPU in default.
  enum SolverMode {
    CPU = 0;
    GPU = 1;
  }
  optional SolverMode solver_mode = 17 [default = GPU];
  // the device_id will that be used in GPU mode. Use device_id = 0 in default.
  optional int32 device_id = 18 [default = 0];
  // If non-negative, the seed with which the Solver will initialize the Caffe
  // random number generator -- useful for reproducible results. Otherwise,
  // (and by default) initialize using a seed derived from the system clock.
  optional int64 random_seed = 20 [default = -1];

  // Solver type
  enum SolverType {
    SGD = 0;
    NESTEROV = 1;
    ADAGRAD = 2;
  }
  optional SolverType solver_type = 30 [default = SGD];
  // numerical stability for AdaGrad
  optional float delta = 31 [default = 1e-8];

  // If true, print information about the state of the net that may help with
  // debugging learning problems.
  optional bool debug_info = 23 [default = false];

  // If false, don't save a snapshot after training finishes.
  optional bool snapshot_after_train = 28 [default = true];
}

// A message that stores the solver snapshots
message SolverState {
  optional int32 iter = 1; // The current iteration
  optional string learned_net = 2; // The file that stores the learned net.
  repeated BlobProto history = 3; // The history for sgd solvers
  optional int32 current_step = 4 [default = 0]; // The current step for learning rate
}

enum Phase {
   TRAIN = 0;
   TEST = 1;
}

message NetState {
  optional Phase phase = 1 [default = TEST];
  optional int32 level = 2 [default = 0];
  repeated string stage = 3;
}

message NetStateRule {
  // Set phase to require the NetState have a particular phase (TRAIN or TEST)
  // to meet this rule.
  optional Phase phase = 1;

  // Set the minimum and/or maximum levels in which the layer should be used.
  // Leave undefined to meet the rule regardless of level.
  optional int32 min_level = 2;
  optional int32 max_level = 3;

  // Customizable sets of stages to include or exclude.
  // The net must have ALL of the specified stages and NONE of the specified
  // "not_stage"s to meet the rule.
  // (Use multiple NetStateRules to specify conjunctions of stages.)
  repeated string stage = 4;
  repeated string not_stage = 5;
}

// Specifies training parameters (multipliers on global learning constants,
// and the name and other settings used for weight sharing).
message ParamSpec {
  // The names of the parameter blobs -- useful for sharing parameters among
  // layers, but never required otherwise.  To share a parameter between two
  // layers, give it a (non-empty) name.
  optional string name = 1;

  // Whether to require shared weights to have the same shape, or just the same
  // count -- defaults to STRICT if unspecified.
  optional DimCheckMode share_mode = 2;
  enum DimCheckMode {
    // STRICT (default) requires that num, channels, height, width each match.
    STRICT = 0;
    // PERMISSIVE requires only the count (num*channels*height*width) to match.
    PERMISSIVE = 1;
  }

  // The multiplier on the global learning rate for this parameter.
  optional float lr_mult = 3 [default = 1.0];

  // The multiplier on the global weight decay for this parameter.
  optional float decay_mult = 4 [default = 1.0];
}

// NOTE
// Update the next available ID when you add a new LayerParameter field.
//
// LayerParameter next available layer-specific ID: 137 (last added: reduction_param)
message LayerParameter {
  optional string name = 1; // the layer name
  optional string type = 2; // the layer type
  repeated string bottom = 3; // the name of each bottom blob
  repeated string top = 4; // the name of each top blob

  // The train / test phase for computation.
  optional Phase phase = 10;

  // The amount of weight to assign each top blob in the objective.
  // Each layer assigns a default value, usually of either 0 or 1,
  // to each top blob.
  repeated float loss_weight = 5;

  // Specifies training parameters (multipliers on global learning constants,
  // and the name and other settings used for weight sharing).
  repeated ParamSpec param = 6;

  // The blobs containing the numeric parameters of the layer.
  repeated BlobProto blobs = 7;
  
  // Specifies on which bottoms the backpropagation should be skipped.
  // The size must be either 0 or equal to the number of bottoms.
  repeated bool propagate_down = 11;

  // Rules controlling whether and when a layer is included in the network,
  // based on the current NetState.  You may specify a non-zero number of rules
  // to include OR exclude, but not both.  If no include or exclude rules are
  // specified, the layer is always included.  If the current NetState meets
  // ANY (i.e., one or more) of the specified rules, the layer is
  // included/excluded.
  repeated NetStateRule include = 8;
  repeated NetStateRule exclude = 9;

  // Parameters for data pre-processing.
  optional TransformationParameter transform_param = 100;

  // Parameters shared by loss layers.
  optional LossParameter loss_param = 101;

  // Layer type-specific parameters.
  //
  // Note: certain layers may have more than one computational engine
  // for their implementation. These layers include an Engine type and
  // engine parameter for selecting the implementation.
  // The default for the engine is set by the ENGINE switch at compile-time.
  optional AccuracyParameter accuracy_param = 102;
  optional ArgMaxParameter argmax_param = 103;
  optional ConcatParameter concat_param = 104;
  optional ContrastiveLossParameter contrastive_loss_param = 105;
  optional ConvolutionParameter convolution_param = 106;
  optional DataParameter data_param = 107;
  optional DropoutParameter dropout_param = 108;
  optional DummyDataParameter dummy_data_param = 109;
  optional EltwiseParameter eltwise_param = 110;
  optional ExpParameter exp_param = 111;
  optional FlattenParameter flatten_param = 135;
  optional HDF5DataParameter hdf5_data_param = 112;
  optional HDF5OutputParameter hdf5_output_param = 113;
  optional HingeLossParameter hinge_loss_param = 114;
  optional ImageDataParameter image_data_param = 115;
  optional InfogainLossParameter infogain_loss_param = 116;
  optional InnerProductParameter inner_product_param = 117;
  optional LogParameter log_param = 134;
  optional LRNParameter lrn_param = 118;
  optional MemoryDataParameter memory_data_param = 119;
  optional MVNParameter mvn_param = 120;
  optional PoolingParameter pooling_param = 121;
  optional PowerParameter power_param = 122;
  optional PReLUParameter prelu_param = 131;
  optional PythonParameter python_param = 130;
  optional ReductionParameter reduction_param = 136;
  optional ReLUParameter relu_param = 123;
  optional ReshapeParameter reshape_param = 133;
  optional SigmoidParameter sigmoid_param = 124;
  optional SoftmaxParameter softmax_param = 125;
  optional SPPParameter spp_param = 132;
  optional SliceParameter slice_param = 126;
  optional TanHParameter tanh_param = 127;
  optional ThresholdParameter threshold_param = 128;
  optional WindowDataParameter window_data_param = 129;
}

// Message that stores parameters used to apply transformation
// to the data layer's data
message TransformationParameter {
  // For data pre-processing, we can do simple scaling and subtracting the
  // data mean, if provided. Note that the mean subtraction is always carried
  // out before scaling.
  optional float scale = 1 [default = 1];
  // Specify if we want to randomly mirror data.
  optional bool mirror = 2 [default = false];
  // Specify if we would like to randomly crop an image.
  optional uint32 crop_size = 3 [default = 0];
  // mean_file and mean_value cannot be specified at the same time
  optional string mean_file = 4;
  // if specified can be repeated once (would substract it from all the channels)
  // or can be repeated the same number of times as channels
  // (would subtract them from the corresponding channel)
  repeated float mean_value = 5;
  // Force the decoded image to have 3 color channels.
  optional bool force_color = 6 [default = false];
  // Force the decoded image to have 1 color channels.
  optional bool force_gray = 7 [default = false];
}

// Message that stores parameters shared by loss layers
message LossParameter {
  // If specified, ignore instances with the given label.
  optional int32 ignore_label = 1;
  // If true, normalize each batch across all instances (including spatial
  // dimesions, but not ignored instances); else, divide by batch size only.
  optional bool normalize = 2 [default = true];
}

// Messages that store parameters used by individual layer types follow, in
// alphabetical order.

message AccuracyParameter {
  // When computing accuracy, count as correct by comparing the true label to
  // the top k scoring classes.  By default, only compare to the top scoring
  // class (i.e. argmax).
  optional uint32 top_k = 1 [default = 1];

  // The "label" axis of the prediction blob, whose argmax corresponds to the
  // predicted label -- may be negative to index from the end (e.g., -1 for the
  // last axis).  For example, if axis == 1 and the predictions are
  // (N x C x H x W), the label blob is expected to contain N*H*W ground truth
  // labels with integer values in {0, 1, ..., C-1}.
  optional int32 axis = 2 [default = 1];

  // If specified, ignore instances with the given label.
  optional int32 ignore_label = 3;
}

message ArgMaxParameter {
  // If true produce pairs (argmax, maxval)
  optional bool out_max_val = 1 [default = false];
  optional uint32 top_k = 2 [default = 1];
}

message ConcatParameter {
  // The axis along which to concatenate -- may be negative to index from the
  // end (e.g., -1 for the last axis).  Other axes must have the
  // same dimension for all the bottom blobs.
  // By default, ConcatLayer concatenates blobs along the "channels" axis (1).
  optional int32 axis = 2 [default = 1];

  // DEPRECATED: alias for "axis" -- does not support negative indexing.
  optional uint32 concat_dim = 1 [default = 1];
}

message ContrastiveLossParameter {
  // margin for dissimilar pair
  optional float margin = 1 [default = 1.0];
  // The first implementation of this cost did not exactly match the cost of
  // Hadsell et al 2006 -- using (margin - d^2) instead of (margin - d)^2.
  // legacy_version = false (the default) uses (margin - d)^2 as proposed in the
  // Hadsell paper. New models should probably use this version.
  // legacy_version = true uses (margin - d^2). This is kept to support /
  // reproduce existing models and results
  optional bool legacy_version = 2 [default = false]; 
}

message ConvolutionParameter {
  optional uint32 num_output = 1; // The number of outputs for the layer
  optional bool bias_term = 2 [default = true]; // whether to have bias terms
  // Pad, kernel size, and stride are all given as a single value for equal
  // dimensions in height and width or as Y, X pairs.
  optional uint32 pad = 3 [default = 0]; // The padding size (equal in Y, X)
  optional uint32 pad_h = 9 [default = 0]; // The padding height
  optional uint32 pad_w = 10 [default = 0]; // The padding width
  optional uint32 kernel_size = 4; // The kernel size (square)
  optional uint32 kernel_h = 11; // The kernel height
  optional uint32 kernel_w = 12; // The kernel width
  optional uint32 group = 5 [default = 1]; // The group size for group conv
  optional uint32 stride = 6 [default = 1]; // The stride (equal in Y, X)
  optional uint32 stride_h = 13; // The stride height
  optional uint32 stride_w = 14; // The stride width
  optional FillerParameter weight_filler = 7; // The filler for the weight
  optional FillerParameter bias_filler = 8; // The filler for the bias
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 15 [default = DEFAULT];
}

message DataParameter {
  enum DB {
    LEVELDB = 0;
    LMDB = 1;
  }
  // Specify the data source.
  optional string source = 1;
  // Specify the batch size.
  optional uint32 batch_size = 4;
  // The rand_skip variable is for the data layer to skip a few data points
  // to avoid all asynchronous sgd clients to start at the same point. The skip
  // point would be set as rand_skip * rand(0,1). Note that rand_skip should not
  // be larger than the number of keys in the database.
  optional uint32 rand_skip = 7 [default = 0];
  optional DB backend = 8 [default = LEVELDB];
  // DEPRECATED. See TransformationParameter. For data pre-processing, we can do
  // simple scaling and subtracting the data mean, if provided. Note that the
  // mean subtraction is always carried out before scaling.
  optional float scale = 2 [default = 1];
  optional string mean_file = 3;
  // DEPRECATED. See TransformationParameter. Specify if we would like to randomly
  // crop an image.
  optional uint32 crop_size = 5 [default = 0];
  // DEPRECATED. See TransformationParameter. Specify if we want to randomly mirror
  // data.
  optional bool mirror = 6 [default = false];
  // Force the encoded image to have 3 color channels
  optional bool force_encoded_color = 9 [default = false];
}

message DropoutParameter {
  optional float dropout_ratio = 1 [default = 0.5]; // dropout ratio
}

// DummyDataLayer fills any number of arbitrarily shaped blobs with random
// (or constant) data generated by "Fillers" (see "message FillerParameter").
message DummyDataParameter {
  // This layer produces N >= 1 top blobs.  DummyDataParameter must specify 1 or N
  // shape fields, and 0, 1 or N data_fillers.
  //
  // If 0 data_fillers are specified, ConstantFiller with a value of 0 is used.
  // If 1 data_filler is specified, it is applied to all top blobs.  If N are
  // specified, the ith is applied to the ith top blob.
  repeated FillerParameter data_filler = 1;
  repeated BlobShape shape = 6;

  // 4D dimensions -- deprecated.  Use "shape" instead.
  repeated uint32 num = 2;
  repeated uint32 channels = 3;
  repeated uint32 height = 4;
  repeated uint32 width = 5;
}

message EltwiseParameter {
  enum EltwiseOp {
    PROD = 0;
    SUM = 1;
    MAX = 2;
  }
  optional EltwiseOp operation = 1 [default = SUM]; // element-wise operation
  repeated float coeff = 2; // blob-wise coefficient for SUM operation

  // Whether to use an asymptotically slower (for >2 inputs) but stabler method
  // of computing the gradient for the PROD operation. (No effect for SUM op.)
  optional bool stable_prod_grad = 3 [default = true];
}

message ExpParameter {
  // ExpLayer computes outputs y = base ^ (shift + scale * x), for base > 0.
  // Or if base is set to the default (-1), base is set to e,
  // so y = exp(shift + scale * x).
  optional float base = 1 [default = -1.0];
  optional float scale = 2 [default = 1.0];
  optional float shift = 3 [default = 0.0];
}

/// Message that stores parameters used by FlattenLayer
message FlattenParameter {
  // The first axis to flatten: all preceding axes are retained in the output.
  // May be negative to index from the end (e.g., -1 for the last axis).
  optional int32 axis = 1 [default = 1];

  // The last axis to flatten: all following axes are retained in the output.
  // May be negative to index from the end (e.g., the default -1 for the last
  // axis).
  optional int32 end_axis = 2 [default = -1];
}

// Message that stores parameters used by HDF5DataLayer
message HDF5DataParameter {
  // Specify the data source.
  optional string source = 1;
  // Specify the batch size.
  optional uint32 batch_size = 2;

  // Specify whether to shuffle the data.
  // If shuffle == true, the ordering of the HDF5 files is shuffled,
  // and the ordering of data within any given HDF5 file is shuffled,
  // but data between different files are not interleaved; all of a file's
  // data are output (in a random order) before moving onto another file.
  optional bool shuffle = 3 [default = false];
}

message HDF5OutputParameter {
  optional string file_name = 1;
}

message HingeLossParameter {
  enum Norm {
    L1 = 1;
    L2 = 2;
  }
  // Specify the Norm to use L1 or L2
  optional Norm norm = 1 [default = L1];
}

message ImageDataParameter {
  // Specify the data source.
  optional string source = 1;
  // Specify the batch size.
  optional uint32 batch_size = 4;
  // The rand_skip variable is for the data layer to skip a few data points
  // to avoid all asynchronous sgd clients to start at the same point. The skip
  // point would be set as rand_skip * rand(0,1). Note that rand_skip should not
  // be larger than the number of keys in the database.
  optional uint32 rand_skip = 7 [default = 0];
  // Whether or not ImageLayer should shuffle the list of files at every epoch.
  optional bool shuffle = 8 [default = false];
  // It will also resize images if new_height or new_width are not zero.
  optional uint32 new_height = 9 [default = 0];
  optional uint32 new_width = 10 [default = 0];
  // Specify if the images are color or gray
  optional bool is_color = 11 [default = true];
  // DEPRECATED. See TransformationParameter. For data pre-processing, we can do
  // simple scaling and subtracting the data mean, if provided. Note that the
  // mean subtraction is always carried out before scaling.
  optional float scale = 2 [default = 1];
  optional string mean_file = 3;
  // DEPRECATED. See TransformationParameter. Specify if we would like to randomly
  // crop an image.
  optional uint32 crop_size = 5 [default = 0];
  // DEPRECATED. See TransformationParameter. Specify if we want to randomly mirror
  // data.
  optional bool mirror = 6 [default = false];
  optional string root_folder = 12 [default = ""];
}

message InfogainLossParameter {
  // Specify the infogain matrix source.
  optional string source = 1;
}

message InnerProductParameter {
  optional uint32 num_output = 1; // The number of outputs for the layer
  optional bool bias_term = 2 [default = true]; // whether to have bias terms
  optional FillerParameter weight_filler = 3; // The filler for the weight
  optional FillerParameter bias_filler = 4; // The filler for the bias

  // The first axis to be lumped into a single inner product computation;
  // all preceding axes are retained in the output.
  // May be negative to index from the end (e.g., -1 for the last axis).
  optional int32 axis = 5 [default = 1];
}

// Message that stores parameters used by LogLayer
message LogParameter {
  // LogLayer computes outputs y = log_base(shift + scale * x), for base > 0.
  // Or if base is set to the default (-1), base is set to e,
  // so y = ln(shift + scale * x) = log_e(shift + scale * x)
  optional float base = 1 [default = -1.0];
  optional float scale = 2 [default = 1.0];
  optional float shift = 3 [default = 0.0];
}

// Message that stores parameters used by LRNLayer
message LRNParameter {
  optional uint32 local_size = 1 [default = 5];
  optional float alpha = 2 [default = 1.];
  optional float beta = 3 [default = 0.75];
  enum NormRegion {
    ACROSS_CHANNELS = 0;
    WITHIN_CHANNEL = 1;
  }
  optional NormRegion norm_region = 4 [default = ACROSS_CHANNELS];
  optional float k = 5 [default = 1.];
}

message MemoryDataParameter {
  optional uint32 batch_size = 1;
  optional uint32 channels = 2;
  optional uint32 height = 3;
  optional uint32 width = 4;
}

message MVNParameter {
  // This parameter can be set to false to normalize mean only
  optional bool normalize_variance = 1 [default = true];

  // This parameter can be set to true to perform DNN-like MVN
  optional bool across_channels = 2 [default = false];

  // Epsilon for not dividing by zero while normalizing variance
  optional float eps = 3 [default = 1e-9];
}

message PoolingParameter {
  enum PoolMethod {
    MAX = 0;
    AVE = 1;
    STOCHASTIC = 2;
  }
  optional PoolMethod pool = 1 [default = MAX]; // The pooling method
  // Pad, kernel size, and stride are all given as a single value for equal
  // dimensions in height and width or as Y, X pairs.
  optional uint32 pad = 4 [default = 0]; // The padding size (equal in Y, X)
  optional uint32 pad_h = 9 [default = 0]; // The padding height
  optional uint32 pad_w = 10 [default = 0]; // The padding width
  optional uint32 kernel_size = 2; // The kernel size (square)
  optional uint32 kernel_h = 5; // The kernel height
  optional uint32 kernel_w = 6; // The kernel width
  optional uint32 stride = 3 [default = 1]; // The stride (equal in Y, X)
  optional uint32 stride_h = 7; // The stride height
  optional uint32 stride_w = 8; // The stride width
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 11 [default = DEFAULT];
  // If global_pooling then it will pool over the size of the bottom by doing
  // kernel_h = bottom->height and kernel_w = bottom->width
  optional bool global_pooling = 12 [default = false];
}

message PowerParameter {
  // PowerLayer computes outputs y = (shift + scale * x) ^ power.
  optional float power = 1 [default = 1.0];
  optional float scale = 2 [default = 1.0];
  optional float shift = 3 [default = 0.0];
}

message PythonParameter {
  optional string module = 1;
  optional string layer = 2;
}

// Message that stores parameters used by ReductionLayer
message ReductionParameter {
  enum ReductionOp {
    SUM = 1;
    ASUM = 2;
    SUMSQ = 3;
    MEAN = 4;
  }

  optional ReductionOp operation = 1 [default = SUM]; // reduction operation

  // The first axis to reduce to a scalar -- may be negative to index from the
  // end (e.g., -1 for the last axis).
  // (Currently, only reduction along ALL "tail" axes is supported; reduction
  // of axis M through N, where N < num_axes - 1, is unsupported.)
  // Suppose we have an n-axis bottom Blob with shape:
  //     (d0, d1, d2, ..., d(m-1), dm, d(m+1), ..., d(n-1)).
  // If axis == m, the output Blob will have shape
  //     (d0, d1, d2, ..., d(m-1)),
  // and the ReductionOp operation is performed (d0 * d1 * d2 * ... * d(m-1))
  // times, each including (dm * d(m+1) * ... * d(n-1)) individual data.
  // If axis == 0 (the default), the output Blob always has the empty shape
  // (count 1), performing reduction across the entire input --
  // often useful for creating new loss functions.
  optional int32 axis = 2 [default = 0];

  optional float coeff = 3 [default = 1.0]; // coefficient for output
}

// Message that stores parameters used by ReLULayer
message ReLUParameter {
  // Allow non-zero slope for negative inputs to speed up optimization
  // Described in:
  // Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). Rectifier nonlinearities
  // improve neural network acoustic models. In ICML Workshop on Deep Learning
  // for Audio, Speech, and Language Processing.
  optional float negative_slope = 1 [default = 0];
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 2 [default = DEFAULT];
}

message ReshapeParameter {
  // Specify the output dimensions. If some of the dimensions are set to 0,
  // the corresponding dimension from the bottom layer is used (unchanged).
  // Exactly one dimension may be set to -1, in which case its value is
  // inferred from the count of the bottom blob and the remaining dimensions.
  // For example, suppose we want to reshape a 2D blob "input" with shape 2 x 8:
  //
  //   layer {
  //     type: "Reshape" bottom: "input" top: "output"
  //     reshape_param { ... }
  //   }
  //
  // If "input" is 2D with shape 2 x 8, then the following reshape_param
  // specifications are all equivalent, producing a 3D blob "output" with shape
  // 2 x 2 x 4:
  //
  //   reshape_param { shape { dim:  2  dim: 2  dim:  4 } }
  //   reshape_param { shape { dim:  0  dim: 2  dim:  4 } }
  //   reshape_param { shape { dim:  0  dim: 2  dim: -1 } }
  //   reshape_param { shape { dim: -1  dim: 0  dim:  2 } }
  //
  optional BlobShape shape = 1;

  // axis and num_axes control the portion of the bottom blob's shape that are
  // replaced by (included in) the reshape. By default (axis == 0 and
  // num_axes == -1), the entire bottom blob shape is included in the reshape,
  // and hence the shape field must specify the entire output shape.
  //
  // axis may be non-zero to retain some portion of the beginning of the input
  // shape (and may be negative to index from the end; e.g., -1 to begin the
  // reshape after the last axis, including nothing in the reshape,
  // -2 to include only the last axis, etc.).
  //
  // For example, suppose "input" is a 2D blob with shape 2 x 8.
  // Then the following ReshapeLayer specifications are all equivalent,
  // producing a blob "output" with shape 2 x 2 x 4:
  //
  //   reshape_param { shape { dim: 2  dim: 2  dim: 4 } }
  //   reshape_param { shape { dim: 2  dim: 4 } axis:  1 }
  //   reshape_param { shape { dim: 2  dim: 4 } axis: -3 }
  //
  // num_axes specifies the extent of the reshape.
  // If num_axes >= 0 (and axis >= 0), the reshape will be performed only on
  // input axes in the range [axis, axis+num_axes].
  // num_axes may also be -1, the default, to include all remaining axes
  // (starting from axis).
  //
  // For example, suppose "input" is a 2D blob with shape 2 x 8.
  // Then the following ReshapeLayer specifications are equivalent,
  // producing a blob "output" with shape 1 x 2 x 8.
  //
  //   reshape_param { shape { dim:  1  dim: 2  dim:  8 } }
  //   reshape_param { shape { dim:  1  dim: 2  }  num_axes: 1 }
  //   reshape_param { shape { dim:  1  }  num_axes: 0 }
  //
  // On the other hand, these would produce output blob shape 2 x 1 x 8:
  //
  //   reshape_param { shape { dim: 2  dim: 1  dim: 8  }  }
  //   reshape_param { shape { dim: 1 }  axis: 1  num_axes: 0 }
  //
  optional int32 axis = 2 [default = 0];
  optional int32 num_axes = 3 [default = -1];
}

message SigmoidParameter {
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 1 [default = DEFAULT];
}

message SliceParameter {
  // The axis along which to slice -- may be negative to index from the end
  // (e.g., -1 for the last axis).
  // By default, SliceLayer concatenates blobs along the "channels" axis (1).
  optional int32 axis = 3 [default = 1];
  repeated uint32 slice_point = 2;

  // DEPRECATED: alias for "axis" -- does not support negative indexing.
  optional uint32 slice_dim = 1 [default = 1];
}

// Message that stores parameters used by SoftmaxLayer, SoftmaxWithLossLayer
message SoftmaxParameter {
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 1 [default = DEFAULT];

  // The axis along which to perform the softmax -- may be negative to index
  // from the end (e.g., -1 for the last axis).
  // Any other axes will be evaluated as independent softmaxes.
  optional int32 axis = 2 [default = 1];
}

message TanHParameter {
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 1 [default = DEFAULT];
}

message ThresholdParameter {
  optional float threshold = 1 [default = 0]; // Strictly positive values
}

message WindowDataParameter {
  // Specify the data source.
  optional string source = 1;
  // For data pre-processing, we can do simple scaling and subtracting the
  // data mean, if provided. Note that the mean subtraction is always carried
  // out before scaling.
  optional float scale = 2 [default = 1];
  optional string mean_file = 3;
  // Specify the batch size.
  optional uint32 batch_size = 4;
  // Specify if we would like to randomly crop an image.
  optional uint32 crop_size = 5 [default = 0];
  // Specify if we want to randomly mirror data.
  optional bool mirror = 6 [default = false];
  // Foreground (object) overlap threshold
  optional float fg_threshold = 7 [default = 0.5];
  // Background (non-object) overlap threshold
  optional float bg_threshold = 8 [default = 0.5];
  // Fraction of batch that should be foreground objects
  optional float fg_fraction = 9 [default = 0.25];
  // Amount of contextual padding to add around a window
  // (used only by the window_data_layer)
  optional uint32 context_pad = 10 [default = 0];
  // Mode for cropping out a detection window
  // warp: cropped window is warped to a fixed size and aspect ratio
  // square: the tightest square around the window is cropped
  optional string crop_mode = 11 [default = "warp"];
  // cache_images: will load all images in memory for faster access
  optional bool cache_images = 12 [default = false];
  // append root_folder to locate images
  optional string root_folder = 13 [default = ""];
}

message SPPParameter {
  enum PoolMethod {
    MAX = 0;
    AVE = 1;
    STOCHASTIC = 2;
  }
  optional uint32 pyramid_height = 1;
  optional PoolMethod pool = 2 [default = MAX]; // The pooling method
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 6 [default = DEFAULT];
}

// DEPRECATED: use LayerParameter.
message V1LayerParameter {
  repeated string bottom = 2;
  repeated string top = 3;
  optional string name = 4;
  repeated NetStateRule include = 32;
  repeated NetStateRule exclude = 33;
  enum LayerType {
    NONE = 0;
    ABSVAL = 35;
    ACCURACY = 1;
    ARGMAX = 30;
    BNLL = 2;
    CONCAT = 3;
    CONTRASTIVE_LOSS = 37;
    CONVOLUTION = 4;
    DATA = 5;
    DECONVOLUTION = 39;
    DROPOUT = 6;
    DUMMY_DATA = 32;
    EUCLIDEAN_LOSS = 7;
    ELTWISE = 25;
    EXP = 38;
    FLATTEN = 8;
    HDF5_DATA = 9;
    HDF5_OUTPUT = 10;
    HINGE_LOSS = 28;
    IM2COL = 11;
    IMAGE_DATA = 12;
    INFOGAIN_LOSS = 13;
    INNER_PRODUCT = 14;
    LRN = 15;
    MEMORY_DATA = 29;
    MULTINOMIAL_LOGISTIC_LOSS = 16;
    MVN = 34;
    POOLING = 17;
    POWER = 26;
    RELU = 18;
    SIGMOID = 19;
    SIGMOID_CROSS_ENTROPY_LOSS = 27;
    SILENCE = 36;
    SOFTMAX = 20;
    SOFTMAX_LOSS = 21;
    SPLIT = 22;
    SLICE = 33;
    TANH = 23;
    WINDOW_DATA = 24;
    THRESHOLD = 31;
  }
  optional LayerType type = 5;
  repeated BlobProto blobs = 6;
  repeated string param = 1001;
  repeated DimCheckMode blob_share_mode = 1002;
  enum DimCheckMode {
    STRICT = 0;
    PERMISSIVE = 1;
  }
  repeated float blobs_lr = 7;
  repeated float weight_decay = 8;
  repeated float loss_weight = 35;
  optional AccuracyParameter accuracy_param = 27;
  optional ArgMaxParameter argmax_param = 23;
  optional ConcatParameter concat_param = 9;
  optional ContrastiveLossParameter contrastive_loss_param = 40;
  optional ConvolutionParameter convolution_param = 10;
  optional DataParameter data_param = 11;
  optional DropoutParameter dropout_param = 12;
  optional DummyDataParameter dummy_data_param = 26;
  optional EltwiseParameter eltwise_param = 24;
  optional ExpParameter exp_param = 41;
  optional HDF5DataParameter hdf5_data_param = 13;
  optional HDF5OutputParameter hdf5_output_param = 14;
  optional HingeLossParameter hinge_loss_param = 29;
  optional ImageDataParameter image_data_param = 15;
  optional InfogainLossParameter infogain_loss_param = 16;
  optional InnerProductParameter inner_product_param = 17;
  optional LRNParameter lrn_param = 18;
  optional MemoryDataParameter memory_data_param = 22;
  optional MVNParameter mvn_param = 34;
  optional PoolingParameter pooling_param = 19;
  optional PowerParameter power_param = 21;
  optional ReLUParameter relu_param = 30;
  optional SigmoidParameter sigmoid_param = 38;
  optional SoftmaxParameter softmax_param = 39;
  optional SliceParameter slice_param = 31;
  optional TanHParameter tanh_param = 37;
  optional ThresholdParameter threshold_param = 25;
  optional WindowDataParameter window_data_param = 20;
  optional TransformationParameter transform_param = 36;
  optional LossParameter loss_param = 42;
  optional V0LayerParameter layer = 1;
}

// DEPRECATED: V0LayerParameter is the old way of specifying layer parameters
// in Caffe.  We keep this message type around for legacy support.
message V0LayerParameter {
  optional string name = 1; // the layer name
  optional string type = 2; // the string to specify the layer type

  // Parameters to specify layers with inner products.
  optional uint32 num_output = 3; // The number of outputs for the layer
  optional bool biasterm = 4 [default = true]; // whether to have bias terms
  optional FillerParameter weight_filler = 5; // The filler for the weight
  optional FillerParameter bias_filler = 6; // The filler for the bias

  optional uint32 pad = 7 [default = 0]; // The padding size
  optional uint32 kernelsize = 8; // The kernel size
  optional uint32 group = 9 [default = 1]; // The group size for group conv
  optional uint32 stride = 10 [default = 1]; // The stride
  enum PoolMethod {
    MAX = 0;
    AVE = 1;
    STOCHASTIC = 2;
  }
  optional PoolMethod pool = 11 [default = MAX]; // The pooling method
  optional float dropout_ratio = 12 [default = 0.5]; // dropout ratio

  optional uint32 local_size = 13 [default = 5]; // for local response norm
  optional float alpha = 14 [default = 1.]; // for local response norm
  optional float beta = 15 [default = 0.75]; // for local response norm
  optional float k = 22 [default = 1.];

  // For data layers, specify the data source
  optional string source = 16;
  // For data pre-processing, we can do simple scaling and subtracting the
  // data mean, if provided. Note that the mean subtraction is always carried
  // out before scaling.
  optional float scale = 17 [default = 1];
  optional string meanfile = 18;
  // For data layers, specify the batch size.
  optional uint32 batchsize = 19;
  // For data layers, specify if we would like to randomly crop an image.
  optional uint32 cropsize = 20 [default = 0];
  // For data layers, specify if we want to randomly mirror data.
  optional bool mirror = 21 [default = false];

  // The blobs containing the numeric parameters of the layer
  repeated BlobProto blobs = 50;
  // The ratio that is multiplied on the global learning rate. If you want to
  // set the learning ratio for one blob, you need to set it for all blobs.
  repeated float blobs_lr = 51;
  // The weight decay that is multiplied on the global weight decay.
  repeated float weight_decay = 52;

  // The rand_skip variable is for the data layer to skip a few data points
  // to avoid all asynchronous sgd clients to start at the same point. The skip
  // point would be set as rand_skip * rand(0,1). Note that rand_skip should not
  // be larger than the number of keys in the database.
  optional uint32 rand_skip = 53 [default = 0];

  // Fields related to detection (det_*)
  // foreground (object) overlap threshold
  optional float det_fg_threshold = 54 [default = 0.5];
  // background (non-object) overlap threshold
  optional float det_bg_threshold = 55 [default = 0.5];
  // Fraction of batch that should be foreground objects
  optional float det_fg_fraction = 56 [default = 0.25];

  // optional bool OBSOLETE_can_clobber = 57 [default = true];

  // Amount of contextual padding to add around a window
  // (used only by the window_data_layer)
  optional uint32 det_context_pad = 58 [default = 0];

  // Mode for cropping out a detection window
  // warp: cropped window is warped to a fixed size and aspect ratio
  // square: the tightest square around the window is cropped
  optional string det_crop_mode = 59 [default = "warp"];

  // For ReshapeLayer, one needs to specify the new dimensions.
  optional int32 new_num = 60 [default = 0];
  optional int32 new_channels = 61 [default = 0];
  optional int32 new_height = 62 [default = 0];
  optional int32 new_width = 63 [default = 0];

  // Whether or not ImageLayer should shuffle the list of files at every epoch.
  // It will also resize images if new_height or new_width are not zero.
  optional bool shuffle_images = 64 [default = false];

  // For ConcatLayer, one needs to specify the dimension for concatenation, and
  // the other dimensions must be the same for all the bottom blobs.
  // By default it will concatenate blobs along the channels dimension.
  optional uint32 concat_dim = 65 [default = 1];

  optional HDF5OutputParameter hdf5_output_param = 1001;
}

message PReLUParameter {
  // Parametric ReLU described in K. He et al, Delving Deep into Rectifiers:
  // Surpassing Human-Level Performance on ImageNet Classification, 2015.

  // Initial value of a_i. Default is a_i=0.25 for all i.
  optional FillerParameter filler = 1;
  // Whether or not slope paramters are shared across channels.
  optional bool channel_shared = 2 [default = false];
}
[root@cobalt proto]# vi caffe.proto 
[root@cobalt proto]# cd ..
[root@cobalt caffe]# ls
blob.cpp        common.cpp            internal_thread.cpp  layers   proto       syncedmem.cpp  util
CMakeLists.txt  data_transformer.cpp  layer_factory.cpp    net.cpp  solver.cpp  test
[root@cobalt caffe]# cd ..
[root@cobalt src]# cd ..
[root@cobalt caffe]# ls
build               CMakeLists.txt   distribute  INSTALL.md       Makefile.config.example  README.md
caffe.cloc          CMakeLists.txt~  docs        LICENSE          matlab                   scripts
cmake               CONTRIBUTORS.md  examples    Makefile         models                   src
CMakeLists_old.txt  data             include     Makefile.config  python                   tools
[root@cobalt caffe]# vi examples/mnist/lenet_solver.prototxt 
[root@cobalt caffe]# vi examples/mnist/lenet_solver.prototxt 
[root@cobalt caffe]# ls
build               CMakeLists.txt   distribute  INSTALL.md       Makefile.config.example  README.md
caffe.cloc          CMakeLists.txt~  docs        LICENSE          matlab                   scripts
cmake               CONTRIBUTORS.md  examples    Makefile         models                   src
CMakeLists_old.txt  data             include     Makefile.config  python                   tools
[root@cobalt caffe]# ./examples/mnist/train_lenet.sh
I0616 22:00:10.623114 14079 caffe.cpp:117] Use CPU.
I0616 22:00:10.623855 14079 caffe.cpp:121] Starting Optimization
I0616 22:00:10.623970 14079 solver.cpp:32] Initializing solver from parameters: 
test_iter: 100
test_interval: 500
base_lr: 0.01
display: 100
max_iter: 10000
lr_policy: "inv"
gamma: 0.0001
power: 0.75
momentum: 0.9
weight_decay: 0.0005
snapshot: 5000
snapshot_prefix: "examples/mnist/lenet"
solver_mode: CPU
net: "examples/mnist/lenet_train_test.prototxt"
I0616 22:00:10.624132 14079 solver.cpp:70] Creating training net from net file: examples/mnist/lenet_train_test.prototxt
I0616 22:00:10.648947 14079 net.cpp:287] The NetState phase (0) differed from the phase (1) specified by a rule in layer mnist
I0616 22:00:10.648988 14079 net.cpp:287] The NetState phase (0) differed from the phase (1) specified by a rule in layer accuracy
I0616 22:00:10.649103 14079 net.cpp:42] Initializing net from parameters: 
name: "LeNet"
state {
  phase: TRAIN
}
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
I0616 22:00:10.649956 14079 layer_factory.hpp:74] Creating layer mnist
I0616 22:00:10.660408 14079 net.cpp:90] Creating Layer mnist
I0616 22:00:10.660454 14079 net.cpp:368] mnist -> data
I0616 22:00:10.661088 14079 net.cpp:368] mnist -> label
I0616 22:00:10.661135 14079 net.cpp:120] Setting up mnist
I0616 22:00:10.661252 14079 db.cpp:34] Opened lmdb examples/mnist/mnist_train_lmdb
I0616 22:00:10.666985 14079 data_layer.cpp:67] output data size: 64,1,28,28
I0616 22:00:10.667191 14079 net.cpp:127] Top shape: 64 1 28 28 (50176)
I0616 22:00:10.667214 14079 net.cpp:127] Top shape: 64 (64)
I0616 22:00:10.667230 14079 layer_factory.hpp:74] Creating layer conv1
I0616 22:00:10.667268 14079 net.cpp:90] Creating Layer conv1
I0616 22:00:10.667294 14079 net.cpp:410] conv1 <- data
I0616 22:00:10.667326 14079 net.cpp:368] conv1 -> conv1
I0616 22:00:10.667367 14079 net.cpp:120] Setting up conv1
I0616 22:00:10.667886 14079 net.cpp:127] Top shape: 64 20 24 24 (737280)
I0616 22:00:10.667918 14079 layer_factory.hpp:74] Creating layer pool1
I0616 22:00:10.667942 14079 net.cpp:90] Creating Layer pool1
I0616 22:00:10.667959 14079 net.cpp:410] pool1 <- conv1
I0616 22:00:10.667976 14079 net.cpp:368] pool1 -> pool1
I0616 22:00:10.667994 14079 net.cpp:120] Setting up pool1
I0616 22:00:10.668025 14079 net.cpp:127] Top shape: 64 20 12 12 (184320)
I0616 22:00:10.668057 14079 layer_factory.hpp:74] Creating layer conv2
I0616 22:00:10.668076 14079 net.cpp:90] Creating Layer conv2
I0616 22:00:10.668088 14079 net.cpp:410] conv2 <- pool1
I0616 22:00:10.668104 14079 net.cpp:368] conv2 -> conv2
I0616 22:00:10.668124 14079 net.cpp:120] Setting up conv2
I0616 22:00:10.668422 14079 net.cpp:127] Top shape: 64 50 8 8 (204800)
I0616 22:00:10.668447 14079 layer_factory.hpp:74] Creating layer pool2
I0616 22:00:10.668463 14079 net.cpp:90] Creating Layer pool2
I0616 22:00:10.668476 14079 net.cpp:410] pool2 <- conv2
I0616 22:00:10.668491 14079 net.cpp:368] pool2 -> pool2
I0616 22:00:10.668510 14079 net.cpp:120] Setting up pool2
I0616 22:00:10.668529 14079 net.cpp:127] Top shape: 64 50 4 4 (51200)
I0616 22:00:10.668542 14079 layer_factory.hpp:74] Creating layer ip1
I0616 22:00:10.668560 14079 net.cpp:90] Creating Layer ip1
I0616 22:00:10.668576 14079 net.cpp:410] ip1 <- pool2
I0616 22:00:10.668592 14079 net.cpp:368] ip1 -> ip1
I0616 22:00:10.668611 14079 net.cpp:120] Setting up ip1
I0616 22:00:10.672966 14079 net.cpp:127] Top shape: 64 500 (32000)
I0616 22:00:10.672991 14079 layer_factory.hpp:74] Creating layer relu1
I0616 22:00:10.673009 14079 net.cpp:90] Creating Layer relu1
I0616 22:00:10.673023 14079 net.cpp:410] relu1 <- ip1
I0616 22:00:10.673038 14079 net.cpp:357] relu1 -> ip1 (in-place)
I0616 22:00:10.673053 14079 net.cpp:120] Setting up relu1
I0616 22:00:10.673380 14079 net.cpp:127] Top shape: 64 500 (32000)
I0616 22:00:10.673413 14079 layer_factory.hpp:74] Creating layer ip2
I0616 22:00:10.673434 14079 net.cpp:90] Creating Layer ip2
I0616 22:00:10.673459 14079 net.cpp:410] ip2 <- ip1
I0616 22:00:10.673475 14079 net.cpp:368] ip2 -> ip2
I0616 22:00:10.673496 14079 net.cpp:120] Setting up ip2
I0616 22:00:10.673579 14079 net.cpp:127] Top shape: 64 10 (640)
I0616 22:00:10.673599 14079 layer_factory.hpp:74] Creating layer loss
I0616 22:00:10.674060 14079 net.cpp:90] Creating Layer loss
I0616 22:00:10.674091 14079 net.cpp:410] loss <- ip2
I0616 22:00:10.674108 14079 net.cpp:410] loss <- label
I0616 22:00:10.674126 14079 net.cpp:368] loss -> loss
I0616 22:00:10.674152 14079 net.cpp:120] Setting up loss
I0616 22:00:10.674173 14079 layer_factory.hpp:74] Creating layer loss
I0616 22:00:10.674208 14079 net.cpp:127] Top shape: (1)
I0616 22:00:10.674224 14079 net.cpp:129]     with loss weight 1
I0616 22:00:10.674259 14079 net.cpp:192] loss needs backward computation.
I0616 22:00:10.674276 14079 net.cpp:192] ip2 needs backward computation.
I0616 22:00:10.674290 14079 net.cpp:192] relu1 needs backward computation.
I0616 22:00:10.674302 14079 net.cpp:192] ip1 needs backward computation.
I0616 22:00:10.674315 14079 net.cpp:192] pool2 needs backward computation.
I0616 22:00:10.674327 14079 net.cpp:192] conv2 needs backward computation.
I0616 22:00:10.674342 14079 net.cpp:192] pool1 needs backward computation.
I0616 22:00:10.674361 14079 net.cpp:192] conv1 needs backward computation.
I0616 22:00:10.674374 14079 net.cpp:194] mnist does not need backward computation.
I0616 22:00:10.674386 14079 net.cpp:235] This network produces output loss
I0616 22:00:10.674406 14079 net.cpp:482] Collecting Learning Rate and Weight Decay.
I0616 22:00:10.674422 14079 net.cpp:247] Network initialization done.
I0616 22:00:10.674434 14079 net.cpp:248] Memory required for data: 5169924
I0616 22:00:10.674787 14079 solver.cpp:154] Creating test net (#0) specified by net file: examples/mnist/lenet_train_test.prototxt
I0616 22:00:10.674828 14079 net.cpp:287] The NetState phase (1) differed from the phase (0) specified by a rule in layer mnist
I0616 22:00:10.674949 14079 net.cpp:42] Initializing net from parameters: 
name: "LeNet"
state {
  phase: TEST
}
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
I0616 22:00:10.675585 14079 layer_factory.hpp:74] Creating layer mnist
I0616 22:00:10.675607 14079 net.cpp:90] Creating Layer mnist
I0616 22:00:10.675621 14079 net.cpp:368] mnist -> data
I0616 22:00:10.675640 14079 net.cpp:368] mnist -> label
I0616 22:00:10.675658 14079 net.cpp:120] Setting up mnist
I0616 22:00:10.675715 14079 db.cpp:34] Opened lmdb examples/mnist/mnist_test_lmdb
I0616 22:00:10.675747 14079 data_layer.cpp:67] output data size: 100,1,28,28
I0616 22:00:10.675907 14079 net.cpp:127] Top shape: 100 1 28 28 (78400)
I0616 22:00:10.675926 14079 net.cpp:127] Top shape: 100 (100)
I0616 22:00:10.675940 14079 layer_factory.hpp:74] Creating layer label_mnist_1_split
I0616 22:00:10.675972 14079 net.cpp:90] Creating Layer label_mnist_1_split
I0616 22:00:10.675987 14079 net.cpp:410] label_mnist_1_split <- label
I0616 22:00:10.676002 14079 net.cpp:368] label_mnist_1_split -> label_mnist_1_split_0
I0616 22:00:10.676028 14079 net.cpp:368] label_mnist_1_split -> label_mnist_1_split_1
I0616 22:00:10.676049 14079 net.cpp:120] Setting up label_mnist_1_split
I0616 22:00:10.676071 14079 net.cpp:127] Top shape: 100 (100)
I0616 22:00:10.676089 14079 net.cpp:127] Top shape: 100 (100)
I0616 22:00:10.676101 14079 layer_factory.hpp:74] Creating layer conv1
I0616 22:00:10.676126 14079 net.cpp:90] Creating Layer conv1
I0616 22:00:10.676143 14079 net.cpp:410] conv1 <- data
I0616 22:00:10.676161 14079 net.cpp:368] conv1 -> conv1
I0616 22:00:10.676179 14079 net.cpp:120] Setting up conv1
I0616 22:00:10.676213 14079 net.cpp:127] Top shape: 100 20 24 24 (1152000)
I0616 22:00:10.676234 14079 layer_factory.hpp:74] Creating layer pool1
I0616 22:00:10.676251 14079 net.cpp:90] Creating Layer pool1
I0616 22:00:10.676265 14079 net.cpp:410] pool1 <- conv1
I0616 22:00:10.676280 14079 net.cpp:368] pool1 -> pool1
I0616 22:00:10.676298 14079 net.cpp:120] Setting up pool1
I0616 22:00:10.676317 14079 net.cpp:127] Top shape: 100 20 12 12 (288000)
I0616 22:00:10.676331 14079 layer_factory.hpp:74] Creating layer conv2
I0616 22:00:10.676353 14079 net.cpp:90] Creating Layer conv2
I0616 22:00:10.676368 14079 net.cpp:410] conv2 <- pool1
I0616 22:00:10.676384 14079 net.cpp:368] conv2 -> conv2
I0616 22:00:10.676403 14079 net.cpp:120] Setting up conv2
I0616 22:00:10.676689 14079 net.cpp:127] Top shape: 100 50 8 8 (320000)
I0616 22:00:10.676718 14079 layer_factory.hpp:74] Creating layer pool2
I0616 22:00:10.676743 14079 net.cpp:90] Creating Layer pool2
I0616 22:00:10.676774 14079 net.cpp:410] pool2 <- conv2
I0616 22:00:10.676791 14079 net.cpp:368] pool2 -> pool2
I0616 22:00:10.676807 14079 net.cpp:120] Setting up pool2
I0616 22:00:10.676826 14079 net.cpp:127] Top shape: 100 50 4 4 (80000)
I0616 22:00:10.676838 14079 layer_factory.hpp:74] Creating layer ip1
I0616 22:00:10.676862 14079 net.cpp:90] Creating Layer ip1
I0616 22:00:10.676877 14079 net.cpp:410] ip1 <- pool2
I0616 22:00:10.676892 14079 net.cpp:368] ip1 -> ip1
I0616 22:00:10.676913 14079 net.cpp:120] Setting up ip1
I0616 22:00:10.681300 14079 net.cpp:127] Top shape: 100 500 (50000)
I0616 22:00:10.681324 14079 layer_factory.hpp:74] Creating layer relu1
I0616 22:00:10.681339 14079 net.cpp:90] Creating Layer relu1
I0616 22:00:10.681360 14079 net.cpp:410] relu1 <- ip1
I0616 22:00:10.681375 14079 net.cpp:357] relu1 -> ip1 (in-place)
I0616 22:00:10.681391 14079 net.cpp:120] Setting up relu1
I0616 22:00:10.681404 14079 net.cpp:127] Top shape: 100 500 (50000)
I0616 22:00:10.681416 14079 layer_factory.hpp:74] Creating layer ip2
I0616 22:00:10.681432 14079 net.cpp:90] Creating Layer ip2
I0616 22:00:10.681442 14079 net.cpp:410] ip2 <- ip1
I0616 22:00:10.681458 14079 net.cpp:368] ip2 -> ip2
I0616 22:00:10.681474 14079 net.cpp:120] Setting up ip2
I0616 22:00:10.681545 14079 net.cpp:127] Top shape: 100 10 (1000)
I0616 22:00:10.681562 14079 layer_factory.hpp:74] Creating layer ip2_ip2_0_split
I0616 22:00:10.681576 14079 net.cpp:90] Creating Layer ip2_ip2_0_split
I0616 22:00:10.681587 14079 net.cpp:410] ip2_ip2_0_split <- ip2
I0616 22:00:10.681602 14079 net.cpp:368] ip2_ip2_0_split -> ip2_ip2_0_split_0
I0616 22:00:10.681617 14079 net.cpp:368] ip2_ip2_0_split -> ip2_ip2_0_split_1
I0616 22:00:10.681632 14079 net.cpp:120] Setting up ip2_ip2_0_split
I0616 22:00:10.681646 14079 net.cpp:127] Top shape: 100 10 (1000)
I0616 22:00:10.681658 14079 net.cpp:127] Top shape: 100 10 (1000)
I0616 22:00:10.681670 14079 layer_factory.hpp:74] Creating layer accuracy
I0616 22:00:10.681686 14079 net.cpp:90] Creating Layer accuracy
I0616 22:00:10.681699 14079 net.cpp:410] accuracy <- ip2_ip2_0_split_0
I0616 22:00:10.681710 14079 net.cpp:410] accuracy <- label_mnist_1_split_0
I0616 22:00:10.681726 14079 net.cpp:368] accuracy -> accuracy
I0616 22:00:10.681741 14079 net.cpp:120] Setting up accuracy
I0616 22:00:10.681759 14079 net.cpp:127] Top shape: (1)
I0616 22:00:10.681772 14079 layer_factory.hpp:74] Creating layer loss
I0616 22:00:10.681785 14079 net.cpp:90] Creating Layer loss
I0616 22:00:10.681797 14079 net.cpp:410] loss <- ip2_ip2_0_split_1
I0616 22:00:10.681808 14079 net.cpp:410] loss <- label_mnist_1_split_1
I0616 22:00:10.681821 14079 net.cpp:368] loss -> loss
I0616 22:00:10.681835 14079 net.cpp:120] Setting up loss
I0616 22:00:10.681849 14079 layer_factory.hpp:74] Creating layer loss
I0616 22:00:10.681872 14079 net.cpp:127] Top shape: (1)
I0616 22:00:10.681885 14079 net.cpp:129]     with loss weight 1
I0616 22:00:10.681901 14079 net.cpp:192] loss needs backward computation.
I0616 22:00:10.681913 14079 net.cpp:194] accuracy does not need backward computation.
I0616 22:00:10.681926 14079 net.cpp:192] ip2_ip2_0_split needs backward computation.
I0616 22:00:10.681936 14079 net.cpp:192] ip2 needs backward computation.
I0616 22:00:10.681947 14079 net.cpp:192] relu1 needs backward computation.
I0616 22:00:10.681958 14079 net.cpp:192] ip1 needs backward computation.
I0616 22:00:10.681969 14079 net.cpp:192] pool2 needs backward computation.
I0616 22:00:10.681980 14079 net.cpp:192] conv2 needs backward computation.
I0616 22:00:10.681993 14079 net.cpp:192] pool1 needs backward computation.
I0616 22:00:10.682003 14079 net.cpp:192] conv1 needs backward computation.
I0616 22:00:10.682015 14079 net.cpp:194] label_mnist_1_split does not need backward computation.
I0616 22:00:10.682027 14079 net.cpp:194] mnist does not need backward computation.
I0616 22:00:10.682039 14079 net.cpp:235] This network produces output accuracy
I0616 22:00:10.682049 14079 net.cpp:235] This network produces output loss
I0616 22:00:10.682070 14079 net.cpp:482] Collecting Learning Rate and Weight Decay.
I0616 22:00:10.682101 14079 net.cpp:247] Network initialization done.
I0616 22:00:10.682113 14079 net.cpp:248] Memory required for data: 8086808
I0616 22:00:10.682165 14079 solver.cpp:42] Solver scaffolding done.
I0616 22:00:10.682196 14079 solver.cpp:226] Solving LeNet
I0616 22:00:10.682209 14079 solver.cpp:227] Learning Rate Policy: inv
I0616 22:00:10.682224 14079 solver.cpp:270] Iteration 0, Testing net (#0)
I0616 22:00:13.506659 14079 solver.cpp:319]     Test net output #0: accuracy = 0.0611
I0616 22:00:13.506728 14079 solver.cpp:319]     Test net output #1: loss = 2.40979 (* 1 = 2.40979 loss)
I0616 22:00:13.552095 14079 solver.cpp:189] Iteration 0, loss = 2.4428
I0616 22:00:13.552146 14079 solver.cpp:204]     Train net output #0: loss = 2.4428 (* 1 = 2.4428 loss)
I0616 22:00:13.552178 14079 solver.cpp:467] Iteration 0, lr = 0.01
I0616 22:00:17.944950 14079 solver.cpp:189] Iteration 100, loss = 0.184262
I0616 22:00:17.945016 14079 solver.cpp:204]     Train net output #0: loss = 0.184262 (* 1 = 0.184262 loss)
I0616 22:00:17.945034 14079 solver.cpp:467] Iteration 100, lr = 0.00992565
I0616 22:00:22.346171 14079 solver.cpp:189] Iteration 200, loss = 0.165761
I0616 22:00:22.346231 14079 solver.cpp:204]     Train net output #0: loss = 0.165761 (* 1 = 0.165761 loss)
I0616 22:00:22.346247 14079 solver.cpp:467] Iteration 200, lr = 0.00985258
I0616 22:00:26.747632 14079 solver.cpp:189] Iteration 300, loss = 0.17561
I0616 22:00:26.747701 14079 solver.cpp:204]     Train net output #0: loss = 0.17561 (* 1 = 0.17561 loss)
I0616 22:00:26.747717 14079 solver.cpp:467] Iteration 300, lr = 0.00978075
I0616 22:00:31.137603 14079 solver.cpp:189] Iteration 400, loss = 0.0733948
I0616 22:00:31.137668 14079 solver.cpp:204]     Train net output #0: loss = 0.0733947 (* 1 = 0.0733947 loss)
I0616 22:00:31.137684 14079 solver.cpp:467] Iteration 400, lr = 0.00971013
I0616 22:00:35.475671 14079 solver.cpp:270] Iteration 500, Testing net (#0)
I0616 22:00:38.238050 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9724
I0616 22:00:38.238116 14079 solver.cpp:319]     Test net output #1: loss = 0.0869109 (* 1 = 0.0869109 loss)
I0616 22:00:38.280961 14079 solver.cpp:189] Iteration 500, loss = 0.120809
I0616 22:00:38.281011 14079 solver.cpp:204]     Train net output #0: loss = 0.120809 (* 1 = 0.120809 loss)
I0616 22:00:38.281025 14079 solver.cpp:467] Iteration 500, lr = 0.00964069
I0616 22:00:42.662447 14079 solver.cpp:189] Iteration 600, loss = 0.0863383
I0616 22:00:42.662766 14079 solver.cpp:204]     Train net output #0: loss = 0.0863382 (* 1 = 0.0863382 loss)
I0616 22:00:42.662799 14079 solver.cpp:467] Iteration 600, lr = 0.0095724
I0616 22:00:47.060544 14079 solver.cpp:189] Iteration 700, loss = 0.121743
I0616 22:00:47.060611 14079 solver.cpp:204]     Train net output #0: loss = 0.121743 (* 1 = 0.121743 loss)
I0616 22:00:47.060627 14079 solver.cpp:467] Iteration 700, lr = 0.00950522
I0616 22:00:51.450695 14079 solver.cpp:189] Iteration 800, loss = 0.248336
I0616 22:00:51.450763 14079 solver.cpp:204]     Train net output #0: loss = 0.248336 (* 1 = 0.248336 loss)
I0616 22:00:51.450778 14079 solver.cpp:467] Iteration 800, lr = 0.00943913
I0616 22:00:55.844264 14079 solver.cpp:189] Iteration 900, loss = 0.170427
I0616 22:00:55.844326 14079 solver.cpp:204]     Train net output #0: loss = 0.170427 (* 1 = 0.170427 loss)
I0616 22:00:55.844342 14079 solver.cpp:467] Iteration 900, lr = 0.00937411
I0616 22:01:00.202163 14079 solver.cpp:270] Iteration 1000, Testing net (#0)
I0616 22:01:02.967190 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9819
I0616 22:01:02.967252 14079 solver.cpp:319]     Test net output #1: loss = 0.0569951 (* 1 = 0.0569951 loss)
I0616 22:01:03.010056 14079 solver.cpp:189] Iteration 1000, loss = 0.0748953
I0616 22:01:03.010108 14079 solver.cpp:204]     Train net output #0: loss = 0.0748953 (* 1 = 0.0748953 loss)
I0616 22:01:03.010123 14079 solver.cpp:467] Iteration 1000, lr = 0.00931012
I0616 22:01:07.392777 14079 solver.cpp:189] Iteration 1100, loss = 0.0102724
I0616 22:01:07.392845 14079 solver.cpp:204]     Train net output #0: loss = 0.0102724 (* 1 = 0.0102724 loss)
I0616 22:01:07.392861 14079 solver.cpp:467] Iteration 1100, lr = 0.00924715
I0616 22:01:11.786947 14079 solver.cpp:189] Iteration 1200, loss = 0.0363824
I0616 22:01:11.787013 14079 solver.cpp:204]     Train net output #0: loss = 0.0363823 (* 1 = 0.0363823 loss)
I0616 22:01:11.787029 14079 solver.cpp:467] Iteration 1200, lr = 0.00918515
I0616 22:01:16.187772 14079 solver.cpp:189] Iteration 1300, loss = 0.00918504
I0616 22:01:16.188036 14079 solver.cpp:204]     Train net output #0: loss = 0.00918494 (* 1 = 0.00918494 loss)
I0616 22:01:16.188058 14079 solver.cpp:467] Iteration 1300, lr = 0.00912412
I0616 22:01:20.587025 14079 solver.cpp:189] Iteration 1400, loss = 0.00847161
I0616 22:01:20.587093 14079 solver.cpp:204]     Train net output #0: loss = 0.0084715 (* 1 = 0.0084715 loss)
I0616 22:01:20.587110 14079 solver.cpp:467] Iteration 1400, lr = 0.00906403
I0616 22:01:24.943120 14079 solver.cpp:270] Iteration 1500, Testing net (#0)
I0616 22:01:27.803514 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9852
I0616 22:01:27.803585 14079 solver.cpp:319]     Test net output #1: loss = 0.0460353 (* 1 = 0.0460353 loss)
I0616 22:01:27.846441 14079 solver.cpp:189] Iteration 1500, loss = 0.0876192
I0616 22:01:27.846501 14079 solver.cpp:204]     Train net output #0: loss = 0.0876191 (* 1 = 0.0876191 loss)
I0616 22:01:27.846518 14079 solver.cpp:467] Iteration 1500, lr = 0.00900485
I0616 22:01:32.266846 14079 solver.cpp:189] Iteration 1600, loss = 0.0941997
I0616 22:01:32.266908 14079 solver.cpp:204]     Train net output #0: loss = 0.0941996 (* 1 = 0.0941996 loss)
I0616 22:01:32.266924 14079 solver.cpp:467] Iteration 1600, lr = 0.00894657
I0616 22:01:36.649116 14079 solver.cpp:189] Iteration 1700, loss = 0.0277912
I0616 22:01:36.649183 14079 solver.cpp:204]     Train net output #0: loss = 0.0277911 (* 1 = 0.0277911 loss)
I0616 22:01:36.649199 14079 solver.cpp:467] Iteration 1700, lr = 0.00888916
I0616 22:01:41.033515 14079 solver.cpp:189] Iteration 1800, loss = 0.0157134
I0616 22:01:41.033578 14079 solver.cpp:204]     Train net output #0: loss = 0.0157133 (* 1 = 0.0157133 loss)
I0616 22:01:41.033594 14079 solver.cpp:467] Iteration 1800, lr = 0.0088326
I0616 22:01:45.557459 14079 solver.cpp:189] Iteration 1900, loss = 0.123466
I0616 22:01:45.557523 14079 solver.cpp:204]     Train net output #0: loss = 0.123466 (* 1 = 0.123466 loss)
I0616 22:01:45.557539 14079 solver.cpp:467] Iteration 1900, lr = 0.00877687
I0616 22:01:49.896354 14079 solver.cpp:270] Iteration 2000, Testing net (#0)
I0616 22:01:52.657608 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9847
I0616 22:01:52.657675 14079 solver.cpp:319]     Test net output #1: loss = 0.047561 (* 1 = 0.047561 loss)
I0616 22:01:52.700870 14079 solver.cpp:189] Iteration 2000, loss = 0.0126202
I0616 22:01:52.700924 14079 solver.cpp:204]     Train net output #0: loss = 0.0126201 (* 1 = 0.0126201 loss)
I0616 22:01:52.700938 14079 solver.cpp:467] Iteration 2000, lr = 0.00872196
I0616 22:01:57.157335 14079 solver.cpp:189] Iteration 2100, loss = 0.0163126
I0616 22:01:57.157408 14079 solver.cpp:204]     Train net output #0: loss = 0.0163125 (* 1 = 0.0163125 loss)
I0616 22:01:57.157424 14079 solver.cpp:467] Iteration 2100, lr = 0.00866784
I0616 22:02:01.539381 14079 solver.cpp:189] Iteration 2200, loss = 0.015406
I0616 22:02:01.539443 14079 solver.cpp:204]     Train net output #0: loss = 0.0154059 (* 1 = 0.0154059 loss)
I0616 22:02:01.539459 14079 solver.cpp:467] Iteration 2200, lr = 0.0086145
I0616 22:02:05.920531 14079 solver.cpp:189] Iteration 2300, loss = 0.101088
I0616 22:02:05.920593 14079 solver.cpp:204]     Train net output #0: loss = 0.101088 (* 1 = 0.101088 loss)
I0616 22:02:05.920609 14079 solver.cpp:467] Iteration 2300, lr = 0.00856192
I0616 22:02:10.299165 14079 solver.cpp:189] Iteration 2400, loss = 0.00933868
I0616 22:02:10.299232 14079 solver.cpp:204]     Train net output #0: loss = 0.00933861 (* 1 = 0.00933861 loss)
I0616 22:02:10.299249 14079 solver.cpp:467] Iteration 2400, lr = 0.00851008
I0616 22:02:14.649572 14079 solver.cpp:270] Iteration 2500, Testing net (#0)
I0616 22:02:17.422283 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9843
I0616 22:02:17.422348 14079 solver.cpp:319]     Test net output #1: loss = 0.0513023 (* 1 = 0.0513023 loss)
I0616 22:02:17.465198 14079 solver.cpp:189] Iteration 2500, loss = 0.0342524
I0616 22:02:17.465248 14079 solver.cpp:204]     Train net output #0: loss = 0.0342524 (* 1 = 0.0342524 loss)
I0616 22:02:17.465263 14079 solver.cpp:467] Iteration 2500, lr = 0.00845897
I0616 22:02:21.843585 14079 solver.cpp:189] Iteration 2600, loss = 0.0788404
I0616 22:02:21.843822 14079 solver.cpp:204]     Train net output #0: loss = 0.0788403 (* 1 = 0.0788403 loss)
I0616 22:02:21.843843 14079 solver.cpp:467] Iteration 2600, lr = 0.00840857
I0616 22:02:26.252071 14079 solver.cpp:189] Iteration 2700, loss = 0.0526081
I0616 22:02:26.252133 14079 solver.cpp:204]     Train net output #0: loss = 0.0526081 (* 1 = 0.0526081 loss)
I0616 22:02:26.252149 14079 solver.cpp:467] Iteration 2700, lr = 0.00835886
I0616 22:02:30.739171 14079 solver.cpp:189] Iteration 2800, loss = 0.00161688
I0616 22:02:30.739236 14079 solver.cpp:204]     Train net output #0: loss = 0.00161684 (* 1 = 0.00161684 loss)
I0616 22:02:30.739253 14079 solver.cpp:467] Iteration 2800, lr = 0.00830984
I0616 22:02:35.117445 14079 solver.cpp:189] Iteration 2900, loss = 0.0092566
I0616 22:02:35.117508 14079 solver.cpp:204]     Train net output #0: loss = 0.00925656 (* 1 = 0.00925656 loss)
I0616 22:02:35.117524 14079 solver.cpp:467] Iteration 2900, lr = 0.00826148
I0616 22:02:39.453488 14079 solver.cpp:270] Iteration 3000, Testing net (#0)
I0616 22:02:42.212765 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9859
I0616 22:02:42.212826 14079 solver.cpp:319]     Test net output #1: loss = 0.042426 (* 1 = 0.042426 loss)
I0616 22:02:42.255643 14079 solver.cpp:189] Iteration 3000, loss = 0.00869635
I0616 22:02:42.255694 14079 solver.cpp:204]     Train net output #0: loss = 0.00869629 (* 1 = 0.00869629 loss)
I0616 22:02:42.255709 14079 solver.cpp:467] Iteration 3000, lr = 0.00821377
I0616 22:02:46.632866 14079 solver.cpp:189] Iteration 3100, loss = 0.0162688
I0616 22:02:46.632926 14079 solver.cpp:204]     Train net output #0: loss = 0.0162688 (* 1 = 0.0162688 loss)
I0616 22:02:46.632941 14079 solver.cpp:467] Iteration 3100, lr = 0.0081667
I0616 22:02:51.012089 14079 solver.cpp:189] Iteration 3200, loss = 0.00667614
I0616 22:02:51.012151 14079 solver.cpp:204]     Train net output #0: loss = 0.00667608 (* 1 = 0.00667608 loss)
I0616 22:02:51.012167 14079 solver.cpp:467] Iteration 3200, lr = 0.00812025
I0616 22:02:55.390403 14079 solver.cpp:189] Iteration 3300, loss = 0.0127498
I0616 22:02:55.390594 14079 solver.cpp:204]     Train net output #0: loss = 0.0127497 (* 1 = 0.0127497 loss)
I0616 22:02:55.390614 14079 solver.cpp:467] Iteration 3300, lr = 0.00807442
I0616 22:02:59.768822 14079 solver.cpp:189] Iteration 3400, loss = 0.00794287
I0616 22:02:59.768884 14079 solver.cpp:204]     Train net output #0: loss = 0.00794282 (* 1 = 0.00794282 loss)
I0616 22:02:59.768900 14079 solver.cpp:467] Iteration 3400, lr = 0.00802918
I0616 22:03:04.200652 14079 solver.cpp:270] Iteration 3500, Testing net (#0)
I0616 22:03:06.982267 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9853
I0616 22:03:06.982329 14079 solver.cpp:319]     Test net output #1: loss = 0.043579 (* 1 = 0.043579 loss)
I0616 22:03:07.025147 14079 solver.cpp:189] Iteration 3500, loss = 0.00414522
I0616 22:03:07.025198 14079 solver.cpp:204]     Train net output #0: loss = 0.00414515 (* 1 = 0.00414515 loss)
I0616 22:03:07.025213 14079 solver.cpp:467] Iteration 3500, lr = 0.00798454
I0616 22:03:11.404958 14079 solver.cpp:189] Iteration 3600, loss = 0.0375424
I0616 22:03:11.405025 14079 solver.cpp:204]     Train net output #0: loss = 0.0375423 (* 1 = 0.0375423 loss)
I0616 22:03:11.405040 14079 solver.cpp:467] Iteration 3600, lr = 0.00794046
I0616 22:03:15.843894 14079 solver.cpp:189] Iteration 3700, loss = 0.017092
I0616 22:03:15.843955 14079 solver.cpp:204]     Train net output #0: loss = 0.017092 (* 1 = 0.017092 loss)
I0616 22:03:15.843971 14079 solver.cpp:467] Iteration 3700, lr = 0.00789695
I0616 22:03:20.277097 14079 solver.cpp:189] Iteration 3800, loss = 0.0100743
I0616 22:03:20.277161 14079 solver.cpp:204]     Train net output #0: loss = 0.0100742 (* 1 = 0.0100742 loss)
I0616 22:03:20.277178 14079 solver.cpp:467] Iteration 3800, lr = 0.007854
I0616 22:03:24.654809 14079 solver.cpp:189] Iteration 3900, loss = 0.0271909
I0616 22:03:24.654876 14079 solver.cpp:204]     Train net output #0: loss = 0.0271908 (* 1 = 0.0271908 loss)
I0616 22:03:24.654893 14079 solver.cpp:467] Iteration 3900, lr = 0.00781158
I0616 22:03:29.118742 14079 solver.cpp:270] Iteration 4000, Testing net (#0)
I0616 22:03:31.877817 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9893
I0616 22:03:31.877877 14079 solver.cpp:319]     Test net output #1: loss = 0.0320494 (* 1 = 0.0320494 loss)
I0616 22:03:31.920606 14079 solver.cpp:189] Iteration 4000, loss = 0.0266625
I0616 22:03:31.920657 14079 solver.cpp:204]     Train net output #0: loss = 0.0266624 (* 1 = 0.0266624 loss)
I0616 22:03:31.920671 14079 solver.cpp:467] Iteration 4000, lr = 0.0077697
I0616 22:03:36.428762 14079 solver.cpp:189] Iteration 4100, loss = 0.0295697
I0616 22:03:36.428825 14079 solver.cpp:204]     Train net output #0: loss = 0.0295696 (* 1 = 0.0295696 loss)
I0616 22:03:36.428841 14079 solver.cpp:467] Iteration 4100, lr = 0.00772833
I0616 22:03:40.875294 14079 solver.cpp:189] Iteration 4200, loss = 0.015864
I0616 22:03:40.875357 14079 solver.cpp:204]     Train net output #0: loss = 0.0158639 (* 1 = 0.0158639 loss)
I0616 22:03:40.875375 14079 solver.cpp:467] Iteration 4200, lr = 0.00768748
I0616 22:03:45.398402 14079 solver.cpp:189] Iteration 4300, loss = 0.040452
I0616 22:03:45.398461 14079 solver.cpp:204]     Train net output #0: loss = 0.0404519 (* 1 = 0.0404519 loss)
I0616 22:03:45.398476 14079 solver.cpp:467] Iteration 4300, lr = 0.00764712
I0616 22:03:49.980964 14079 solver.cpp:189] Iteration 4400, loss = 0.0178817
I0616 22:03:49.981025 14079 solver.cpp:204]     Train net output #0: loss = 0.0178816 (* 1 = 0.0178816 loss)
I0616 22:03:49.981040 14079 solver.cpp:467] Iteration 4400, lr = 0.00760726
I0616 22:03:54.526566 14079 solver.cpp:270] Iteration 4500, Testing net (#0)
I0616 22:03:57.368573 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9881
I0616 22:03:57.368634 14079 solver.cpp:319]     Test net output #1: loss = 0.0353777 (* 1 = 0.0353777 loss)
I0616 22:03:57.411927 14079 solver.cpp:189] Iteration 4500, loss = 0.00736429
I0616 22:03:57.411981 14079 solver.cpp:204]     Train net output #0: loss = 0.0073642 (* 1 = 0.0073642 loss)
I0616 22:03:57.411995 14079 solver.cpp:467] Iteration 4500, lr = 0.00756788
I0616 22:04:01.899534 14079 solver.cpp:189] Iteration 4600, loss = 0.0181392
I0616 22:04:01.899678 14079 solver.cpp:204]     Train net output #0: loss = 0.0181391 (* 1 = 0.0181391 loss)
I0616 22:04:01.899694 14079 solver.cpp:467] Iteration 4600, lr = 0.00752897
I0616 22:04:06.481235 14079 solver.cpp:189] Iteration 4700, loss = 0.00695883
I0616 22:04:06.481307 14079 solver.cpp:204]     Train net output #0: loss = 0.00695875 (* 1 = 0.00695875 loss)
I0616 22:04:06.481338 14079 solver.cpp:467] Iteration 4700, lr = 0.00749052
I0616 22:04:10.953300 14079 solver.cpp:189] Iteration 4800, loss = 0.0127832
I0616 22:04:10.953387 14079 solver.cpp:204]     Train net output #0: loss = 0.0127831 (* 1 = 0.0127831 loss)
I0616 22:04:10.953404 14079 solver.cpp:467] Iteration 4800, lr = 0.00745253
I0616 22:04:15.348811 14079 solver.cpp:189] Iteration 4900, loss = 0.00780487
I0616 22:04:15.348870 14079 solver.cpp:204]     Train net output #0: loss = 0.00780478 (* 1 = 0.00780478 loss)
I0616 22:04:15.348886 14079 solver.cpp:467] Iteration 4900, lr = 0.00741498
I0616 22:04:19.698024 14079 solver.cpp:337] Snapshotting to examples/mnist/lenet_iter_5000.caffemodel
I0616 22:04:19.701272 14079 solver.cpp:345] Snapshotting solver state to examples/mnist/lenet_iter_5000.solverstate
I0616 22:04:19.703429 14079 solver.cpp:270] Iteration 5000, Testing net (#0)
I0616 22:04:22.472481 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9889
I0616 22:04:22.472543 14079 solver.cpp:319]     Test net output #1: loss = 0.0325356 (* 1 = 0.0325356 loss)
I0616 22:04:22.515069 14079 solver.cpp:189] Iteration 5000, loss = 0.0545811
I0616 22:04:22.515123 14079 solver.cpp:204]     Train net output #0: loss = 0.054581 (* 1 = 0.054581 loss)
I0616 22:04:22.515138 14079 solver.cpp:467] Iteration 5000, lr = 0.00737788
I0616 22:04:26.898404 14079 solver.cpp:189] Iteration 5100, loss = 0.0146689
I0616 22:04:26.898471 14079 solver.cpp:204]     Train net output #0: loss = 0.0146688 (* 1 = 0.0146688 loss)
I0616 22:04:26.898488 14079 solver.cpp:467] Iteration 5100, lr = 0.0073412
I0616 22:04:31.276085 14079 solver.cpp:189] Iteration 5200, loss = 0.00860133
I0616 22:04:31.276147 14079 solver.cpp:204]     Train net output #0: loss = 0.00860125 (* 1 = 0.00860125 loss)
I0616 22:04:31.276162 14079 solver.cpp:467] Iteration 5200, lr = 0.00730495
I0616 22:04:35.652767 14079 solver.cpp:189] Iteration 5300, loss = 0.00179856
I0616 22:04:35.653039 14079 solver.cpp:204]     Train net output #0: loss = 0.00179849 (* 1 = 0.00179849 loss)
I0616 22:04:35.653071 14079 solver.cpp:467] Iteration 5300, lr = 0.00726911
I0616 22:04:40.033272 14079 solver.cpp:189] Iteration 5400, loss = 0.0125467
I0616 22:04:40.033340 14079 solver.cpp:204]     Train net output #0: loss = 0.0125467 (* 1 = 0.0125467 loss)
I0616 22:04:40.033359 14079 solver.cpp:467] Iteration 5400, lr = 0.00723368
I0616 22:04:44.366755 14079 solver.cpp:270] Iteration 5500, Testing net (#0)
I0616 22:04:47.123726 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9885
I0616 22:04:47.123788 14079 solver.cpp:319]     Test net output #1: loss = 0.0345866 (* 1 = 0.0345866 loss)
I0616 22:04:47.166419 14079 solver.cpp:189] Iteration 5500, loss = 0.0055787
I0616 22:04:47.166472 14079 solver.cpp:204]     Train net output #0: loss = 0.00557864 (* 1 = 0.00557864 loss)
I0616 22:04:47.166486 14079 solver.cpp:467] Iteration 5500, lr = 0.00719865
I0616 22:04:51.541872 14079 solver.cpp:189] Iteration 5600, loss = 0.000999174
I0616 22:04:51.541935 14079 solver.cpp:204]     Train net output #0: loss = 0.00099912 (* 1 = 0.00099912 loss)
I0616 22:04:51.541950 14079 solver.cpp:467] Iteration 5600, lr = 0.00716402
I0616 22:04:55.917094 14079 solver.cpp:189] Iteration 5700, loss = 0.00361763
I0616 22:04:55.917157 14079 solver.cpp:204]     Train net output #0: loss = 0.00361757 (* 1 = 0.00361757 loss)
I0616 22:04:55.917173 14079 solver.cpp:467] Iteration 5700, lr = 0.00712977
I0616 22:05:00.291911 14079 solver.cpp:189] Iteration 5800, loss = 0.0334266
I0616 22:05:00.291980 14079 solver.cpp:204]     Train net output #0: loss = 0.0334266 (* 1 = 0.0334266 loss)
I0616 22:05:00.291996 14079 solver.cpp:467] Iteration 5800, lr = 0.0070959
I0616 22:05:04.668661 14079 solver.cpp:189] Iteration 5900, loss = 0.00387859
I0616 22:05:04.668725 14079 solver.cpp:204]     Train net output #0: loss = 0.00387855 (* 1 = 0.00387855 loss)
I0616 22:05:04.668740 14079 solver.cpp:467] Iteration 5900, lr = 0.0070624
I0616 22:05:08.999896 14079 solver.cpp:270] Iteration 6000, Testing net (#0)
I0616 22:05:11.754583 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9906
I0616 22:05:11.754645 14079 solver.cpp:319]     Test net output #1: loss = 0.0301621 (* 1 = 0.0301621 loss)
I0616 22:05:11.797245 14079 solver.cpp:189] Iteration 6000, loss = 0.0054838
I0616 22:05:11.797299 14079 solver.cpp:204]     Train net output #0: loss = 0.00548376 (* 1 = 0.00548376 loss)
I0616 22:05:11.797314 14079 solver.cpp:467] Iteration 6000, lr = 0.00702927
I0616 22:05:16.171581 14079 solver.cpp:189] Iteration 6100, loss = 0.00264483
I0616 22:05:16.171643 14079 solver.cpp:204]     Train net output #0: loss = 0.00264478 (* 1 = 0.00264478 loss)
I0616 22:05:16.171656 14079 solver.cpp:467] Iteration 6100, lr = 0.0069965
I0616 22:05:20.545737 14079 solver.cpp:189] Iteration 6200, loss = 0.0104982
I0616 22:05:20.545797 14079 solver.cpp:204]     Train net output #0: loss = 0.0104981 (* 1 = 0.0104981 loss)
I0616 22:05:20.545812 14079 solver.cpp:467] Iteration 6200, lr = 0.00696408
I0616 22:05:24.919777 14079 solver.cpp:189] Iteration 6300, loss = 0.00939473
I0616 22:05:24.919838 14079 solver.cpp:204]     Train net output #0: loss = 0.00939467 (* 1 = 0.00939467 loss)
I0616 22:05:24.919854 14079 solver.cpp:467] Iteration 6300, lr = 0.00693201
I0616 22:05:29.296355 14079 solver.cpp:189] Iteration 6400, loss = 0.00628343
I0616 22:05:29.296434 14079 solver.cpp:204]     Train net output #0: loss = 0.00628336 (* 1 = 0.00628336 loss)
I0616 22:05:29.296468 14079 solver.cpp:467] Iteration 6400, lr = 0.00690029
I0616 22:05:33.628538 14079 solver.cpp:270] Iteration 6500, Testing net (#0)
I0616 22:05:36.381885 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9897
I0616 22:05:36.381942 14079 solver.cpp:319]     Test net output #1: loss = 0.0313139 (* 1 = 0.0313139 loss)
I0616 22:05:36.424768 14079 solver.cpp:189] Iteration 6500, loss = 0.0108798
I0616 22:05:36.424824 14079 solver.cpp:204]     Train net output #0: loss = 0.0108797 (* 1 = 0.0108797 loss)
I0616 22:05:36.424839 14079 solver.cpp:467] Iteration 6500, lr = 0.0068689
I0616 22:05:40.807293 14079 solver.cpp:189] Iteration 6600, loss = 0.0466152
I0616 22:05:40.807534 14079 solver.cpp:204]     Train net output #0: loss = 0.0466151 (* 1 = 0.0466151 loss)
I0616 22:05:40.807572 14079 solver.cpp:467] Iteration 6600, lr = 0.00683784
I0616 22:05:45.190759 14079 solver.cpp:189] Iteration 6700, loss = 0.00752358
I0616 22:05:45.190822 14079 solver.cpp:204]     Train net output #0: loss = 0.00752351 (* 1 = 0.00752351 loss)
I0616 22:05:45.190839 14079 solver.cpp:467] Iteration 6700, lr = 0.00680711
I0616 22:05:49.564929 14079 solver.cpp:189] Iteration 6800, loss = 0.00309714
I0616 22:05:49.564990 14079 solver.cpp:204]     Train net output #0: loss = 0.00309707 (* 1 = 0.00309707 loss)
I0616 22:05:49.565006 14079 solver.cpp:467] Iteration 6800, lr = 0.0067767
I0616 22:05:53.942996 14079 solver.cpp:189] Iteration 6900, loss = 0.00889065
I0616 22:05:53.943054 14079 solver.cpp:204]     Train net output #0: loss = 0.00889058 (* 1 = 0.00889058 loss)
I0616 22:05:53.943070 14079 solver.cpp:467] Iteration 6900, lr = 0.0067466
I0616 22:05:58.285661 14079 solver.cpp:270] Iteration 7000, Testing net (#0)
I0616 22:06:01.039131 14079 solver.cpp:319]     Test net output #0: accuracy = 0.99
I0616 22:06:01.039199 14079 solver.cpp:319]     Test net output #1: loss = 0.0303078 (* 1 = 0.0303078 loss)
I0616 22:06:01.082726 14079 solver.cpp:189] Iteration 7000, loss = 0.00682619
I0616 22:06:01.082779 14079 solver.cpp:204]     Train net output #0: loss = 0.00682611 (* 1 = 0.00682611 loss)
I0616 22:06:01.082794 14079 solver.cpp:467] Iteration 7000, lr = 0.00671681
I0616 22:06:05.453286 14079 solver.cpp:189] Iteration 7100, loss = 0.0483374
I0616 22:06:05.453352 14079 solver.cpp:204]     Train net output #0: loss = 0.0483374 (* 1 = 0.0483374 loss)
I0616 22:06:05.453368 14079 solver.cpp:467] Iteration 7100, lr = 0.00668733
I0616 22:06:09.824206 14079 solver.cpp:189] Iteration 7200, loss = 0.00452895
I0616 22:06:09.824275 14079 solver.cpp:204]     Train net output #0: loss = 0.00452887 (* 1 = 0.00452887 loss)
I0616 22:06:09.824290 14079 solver.cpp:467] Iteration 7200, lr = 0.00665815
I0616 22:06:14.193665 14079 solver.cpp:189] Iteration 7300, loss = 0.0251593
I0616 22:06:14.193827 14079 solver.cpp:204]     Train net output #0: loss = 0.0251592 (* 1 = 0.0251592 loss)
I0616 22:06:14.193850 14079 solver.cpp:467] Iteration 7300, lr = 0.00662927
I0616 22:06:18.563684 14079 solver.cpp:189] Iteration 7400, loss = 0.0071457
I0616 22:06:18.563750 14079 solver.cpp:204]     Train net output #0: loss = 0.00714562 (* 1 = 0.00714562 loss)
I0616 22:06:18.563766 14079 solver.cpp:467] Iteration 7400, lr = 0.00660067
I0616 22:06:22.890676 14079 solver.cpp:270] Iteration 7500, Testing net (#0)
I0616 22:06:25.644548 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9899
I0616 22:06:25.644611 14079 solver.cpp:319]     Test net output #1: loss = 0.0324398 (* 1 = 0.0324398 loss)
I0616 22:06:25.687625 14079 solver.cpp:189] Iteration 7500, loss = 0.00205004
I0616 22:06:25.687680 14079 solver.cpp:204]     Train net output #0: loss = 0.00204995 (* 1 = 0.00204995 loss)
I0616 22:06:25.687693 14079 solver.cpp:467] Iteration 7500, lr = 0.00657236
I0616 22:06:30.062489 14079 solver.cpp:189] Iteration 7600, loss = 0.00559253
I0616 22:06:30.062559 14079 solver.cpp:204]     Train net output #0: loss = 0.00559245 (* 1 = 0.00559245 loss)
I0616 22:06:30.062575 14079 solver.cpp:467] Iteration 7600, lr = 0.00654433
I0616 22:06:34.431587 14079 solver.cpp:189] Iteration 7700, loss = 0.0333867
I0616 22:06:34.431648 14079 solver.cpp:204]     Train net output #0: loss = 0.0333866 (* 1 = 0.0333866 loss)
I0616 22:06:34.431663 14079 solver.cpp:467] Iteration 7700, lr = 0.00651658
I0616 22:06:38.832643 14079 solver.cpp:189] Iteration 7800, loss = 0.00178646
I0616 22:06:38.832710 14079 solver.cpp:204]     Train net output #0: loss = 0.00178638 (* 1 = 0.00178638 loss)
I0616 22:06:38.832726 14079 solver.cpp:467] Iteration 7800, lr = 0.00648911
I0616 22:06:43.258127 14079 solver.cpp:189] Iteration 7900, loss = 0.00506939
I0616 22:06:43.258195 14079 solver.cpp:204]     Train net output #0: loss = 0.00506931 (* 1 = 0.00506931 loss)
I0616 22:06:43.258213 14079 solver.cpp:467] Iteration 7900, lr = 0.0064619
I0616 22:06:47.613795 14079 solver.cpp:270] Iteration 8000, Testing net (#0)
I0616 22:06:50.398463 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9906
I0616 22:06:50.398550 14079 solver.cpp:319]     Test net output #1: loss = 0.0295985 (* 1 = 0.0295985 loss)
I0616 22:06:50.442400 14079 solver.cpp:189] Iteration 8000, loss = 0.00598017
I0616 22:06:50.442461 14079 solver.cpp:204]     Train net output #0: loss = 0.00598008 (* 1 = 0.00598008 loss)
I0616 22:06:50.442489 14079 solver.cpp:467] Iteration 8000, lr = 0.00643496
I0616 22:06:54.928129 14079 solver.cpp:189] Iteration 8100, loss = 0.0158931
I0616 22:06:54.928194 14079 solver.cpp:204]     Train net output #0: loss = 0.0158931 (* 1 = 0.0158931 loss)
I0616 22:06:54.928208 14079 solver.cpp:467] Iteration 8100, lr = 0.00640827
I0616 22:06:59.378232 14079 solver.cpp:189] Iteration 8200, loss = 0.00911264
I0616 22:06:59.378289 14079 solver.cpp:204]     Train net output #0: loss = 0.00911256 (* 1 = 0.00911256 loss)
I0616 22:06:59.378304 14079 solver.cpp:467] Iteration 8200, lr = 0.00638185
I0616 22:07:03.786613 14079 solver.cpp:189] Iteration 8300, loss = 0.0233701
I0616 22:07:03.786695 14079 solver.cpp:204]     Train net output #0: loss = 0.02337 (* 1 = 0.02337 loss)
I0616 22:07:03.786710 14079 solver.cpp:467] Iteration 8300, lr = 0.00635567
I0616 22:07:08.249948 14079 solver.cpp:189] Iteration 8400, loss = 0.00624898
I0616 22:07:08.250010 14079 solver.cpp:204]     Train net output #0: loss = 0.0062489 (* 1 = 0.0062489 loss)
I0616 22:07:08.250025 14079 solver.cpp:467] Iteration 8400, lr = 0.00632975
I0616 22:07:12.621364 14079 solver.cpp:270] Iteration 8500, Testing net (#0)
I0616 22:07:15.403403 14079 solver.cpp:319]     Test net output #0: accuracy = 0.991
I0616 22:07:15.403466 14079 solver.cpp:319]     Test net output #1: loss = 0.0300884 (* 1 = 0.0300884 loss)
I0616 22:07:15.446220 14079 solver.cpp:189] Iteration 8500, loss = 0.00747168
I0616 22:07:15.446270 14079 solver.cpp:204]     Train net output #0: loss = 0.00747159 (* 1 = 0.00747159 loss)
I0616 22:07:15.446285 14079 solver.cpp:467] Iteration 8500, lr = 0.00630407
I0616 22:07:19.876158 14079 solver.cpp:189] Iteration 8600, loss = 0.00169817
I0616 22:07:19.876332 14079 solver.cpp:204]     Train net output #0: loss = 0.00169809 (* 1 = 0.00169809 loss)
I0616 22:07:19.876356 14079 solver.cpp:467] Iteration 8600, lr = 0.00627864
I0616 22:07:24.268193 14079 solver.cpp:189] Iteration 8700, loss = 0.00283161
I0616 22:07:24.268267 14079 solver.cpp:204]     Train net output #0: loss = 0.00283153 (* 1 = 0.00283153 loss)
I0616 22:07:24.268299 14079 solver.cpp:467] Iteration 8700, lr = 0.00625344
I0616 22:07:28.707949 14079 solver.cpp:189] Iteration 8800, loss = 0.00165884
I0616 22:07:28.708010 14079 solver.cpp:204]     Train net output #0: loss = 0.00165876 (* 1 = 0.00165876 loss)
I0616 22:07:28.708026 14079 solver.cpp:467] Iteration 8800, lr = 0.00622847
I0616 22:07:33.157737 14079 solver.cpp:189] Iteration 8900, loss = 0.00148175
I0616 22:07:33.157805 14079 solver.cpp:204]     Train net output #0: loss = 0.00148167 (* 1 = 0.00148167 loss)
I0616 22:07:33.157821 14079 solver.cpp:467] Iteration 8900, lr = 0.00620374
I0616 22:07:37.506036 14079 solver.cpp:270] Iteration 9000, Testing net (#0)
I0616 22:07:40.274962 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9899
I0616 22:07:40.275030 14079 solver.cpp:319]     Test net output #1: loss = 0.0303387 (* 1 = 0.0303387 loss)
I0616 22:07:40.317728 14079 solver.cpp:189] Iteration 9000, loss = 0.0175141
I0616 22:07:40.317776 14079 solver.cpp:204]     Train net output #0: loss = 0.017514 (* 1 = 0.017514 loss)
I0616 22:07:40.317790 14079 solver.cpp:467] Iteration 9000, lr = 0.00617924
I0616 22:07:44.687428 14079 solver.cpp:189] Iteration 9100, loss = 0.00611566
I0616 22:07:44.687490 14079 solver.cpp:204]     Train net output #0: loss = 0.00611557 (* 1 = 0.00611557 loss)
I0616 22:07:44.687505 14079 solver.cpp:467] Iteration 9100, lr = 0.00615496
I0616 22:07:49.056314 14079 solver.cpp:189] Iteration 9200, loss = 0.00394099
I0616 22:07:49.056380 14079 solver.cpp:204]     Train net output #0: loss = 0.0039409 (* 1 = 0.0039409 loss)
I0616 22:07:49.056396 14079 solver.cpp:467] Iteration 9200, lr = 0.0061309
I0616 22:07:53.424993 14079 solver.cpp:189] Iteration 9300, loss = 0.00848908
I0616 22:07:53.425199 14079 solver.cpp:204]     Train net output #0: loss = 0.00848899 (* 1 = 0.00848899 loss)
I0616 22:07:53.425222 14079 solver.cpp:467] Iteration 9300, lr = 0.00610706
I0616 22:07:57.794055 14079 solver.cpp:189] Iteration 9400, loss = 0.027366
I0616 22:07:57.794117 14079 solver.cpp:204]     Train net output #0: loss = 0.0273659 (* 1 = 0.0273659 loss)
I0616 22:07:57.794132 14079 solver.cpp:467] Iteration 9400, lr = 0.00608343
I0616 22:08:02.120028 14079 solver.cpp:270] Iteration 9500, Testing net (#0)
I0616 22:08:04.871414 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9879
I0616 22:08:04.871477 14079 solver.cpp:319]     Test net output #1: loss = 0.0381973 (* 1 = 0.0381973 loss)
I0616 22:08:04.914105 14079 solver.cpp:189] Iteration 9500, loss = 0.00422479
I0616 22:08:04.914155 14079 solver.cpp:204]     Train net output #0: loss = 0.0042247 (* 1 = 0.0042247 loss)
I0616 22:08:04.914170 14079 solver.cpp:467] Iteration 9500, lr = 0.00606002
I0616 22:08:09.281563 14079 solver.cpp:189] Iteration 9600, loss = 0.00254237
I0616 22:08:09.281626 14079 solver.cpp:204]     Train net output #0: loss = 0.00254228 (* 1 = 0.00254228 loss)
I0616 22:08:09.281641 14079 solver.cpp:467] Iteration 9600, lr = 0.00603682
I0616 22:08:13.648224 14079 solver.cpp:189] Iteration 9700, loss = 0.00231429
I0616 22:08:13.648286 14079 solver.cpp:204]     Train net output #0: loss = 0.0023142 (* 1 = 0.0023142 loss)
I0616 22:08:13.648301 14079 solver.cpp:467] Iteration 9700, lr = 0.00601382
I0616 22:08:18.015012 14079 solver.cpp:189] Iteration 9800, loss = 0.0125526
I0616 22:08:18.015072 14079 solver.cpp:204]     Train net output #0: loss = 0.0125525 (* 1 = 0.0125525 loss)
I0616 22:08:18.015087 14079 solver.cpp:467] Iteration 9800, lr = 0.00599102
I0616 22:08:22.381464 14079 solver.cpp:189] Iteration 9900, loss = 0.00576042
I0616 22:08:22.381525 14079 solver.cpp:204]     Train net output #0: loss = 0.00576032 (* 1 = 0.00576032 loss)
I0616 22:08:22.381539 14079 solver.cpp:467] Iteration 9900, lr = 0.00596843
I0616 22:08:26.708294 14079 solver.cpp:337] Snapshotting to examples/mnist/lenet_iter_10000.caffemodel
I0616 22:08:26.711561 14079 solver.cpp:345] Snapshotting solver state to examples/mnist/lenet_iter_10000.solverstate
I0616 22:08:26.731887 14079 solver.cpp:252] Iteration 10000, loss = 0.00286832
I0616 22:08:26.731935 14079 solver.cpp:270] Iteration 10000, Testing net (#0)
I0616 22:08:29.485281 14079 solver.cpp:319]     Test net output #0: accuracy = 0.9908
I0616 22:08:29.485352 14079 solver.cpp:319]     Test net output #1: loss = 0.0286834 (* 1 = 0.0286834 loss)
I0616 22:08:29.485363 14079 solver.cpp:257] Optimization Done.
I0616 22:08:29.485370 14079 caffe.cpp:134] Optimization Done.
[root@cobalt caffe]# 

training the lenet using mnist  dataset
