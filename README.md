# Joint Unsupervised Learning (JULE) of Deep Representations and Image Clusters.

### Overview

This project is a Torch implementation for our CVPR 2016 [paper](https://arxiv.org/abs/1604.03628), which performs jointly unsupervised learning of deep CNN and image clusters. The intuition behind this is that better image representation will facilitate clustering, while better clustering results will help representation learning. Given a unlabeled dataset, it will iteratively learn CNN parameters unsupervisedly and cluster images.

### Disclaimer
This is a torch version reimplementation to the code used in our CVPR paper. There is a slight difference between the code used to report the results in our paper. The Caffe version code can be found [here](https://github.com/jwyang/JULE-Caffe).

### License

This code is released under the MIT License (refer to the LICENSE file for details).

### Citation
If you find our code is useful in your researches, please consider citing:

    @inproceedings{yangCVPR2016joint,
        Author = {Yang, Jianwei and Parikh, Devi and Batra, Dhruv},
        Title = {Joint Unsupervised Learning of Deep Representations and Image Clusters},
        Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        Year = {2016}
    }

### Dependencies

1. [Torch](http://torch.ch/). Install Torch by:

   ```bash
   $ curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
   $ git clone https://github.com/torch/distro.git ~/torch --recursive
   $ cd ~/torch; 
   $ ./install.sh      # and enter "yes" at the end to modify your bashrc
   $ source ~/.bashrc
   ```

   After installing torch, you may also need install some packages using [LuaRocks](https://luarocks.org/):

   ```bash
   $ luarocks install nn
   $ luarocks install image 
   ```

   It is preferred to run the code on GPU. Thus you need to install cunn:

   ```bash
   $ luarocks install cunn
   ```

2. [lua-knn](https://github.com/Saulzar/lua-knn). It is used to compute the distance between neighbor samples. Go into the folder, and then compile it manually. The steps are:

   ```bash
   $ mkdir build && cd build
   $ cmake -D CUDA_TOOLKIT_ROOT_DIR=/your/cuda-toolkit/dir ..
   $ make
   ```
   You will see libknn.so in the same folder. Then copy it to folder *torch/install/lib/lua/5.1*, and copy init.lua to folder *torch/install/share/lua/5.1/knn*. Make a new folder knn if you cannot find it.

Typically, you can run our code after installing the above two packages. Please let me know if error occurs.

### Train model

1. It is very simple to run the code for training model. For example, if you want to train on *USPS* dataset, you can run:

   ```bash
   $ th train.lua -dataset USPS -eta 0.9
   ```

   **Note that it runs on fast mode by default.** You can change it to regular mode by setting "-use_fast 0". In the above command, eta is the unfolding rate. For face dataset, we recommand 0.2, while for other datasets, it is set to 0.9 to save training time. During training, you will see the normalize mutual information (NMI) for the clustering results.

2. You can train multiple models in parallel by:

   ```bash
   $ th train.lua -dataset USPS -eta 0.9 -num_nets 5
   ```
   By this way, you weill get 5 different models, and thus 5 possible different results. Statistics such as mean and stddev can be computed on these results.

3. You can also get the clustering performance when using raw image data and random CNN by
   ```bash
   $ th train.lua -dataset USPS -eta 0.9 -updateCNN 0
   ```

4. You can also change other hyper parameters for model training, such as K_s, K_c, number of epochs in each partial unrolled period, etc.

### Datasets

We upload six small datasets: COIL-20, USPS, MNIST-test, CMU-PIE, FRGC, UMist. The other large datasets, COIL-100, MNIST-full and YTF can be found in my google drive [here](https://drive.google.com/folderview?id=0B9J-9A2jotGRT25vSDhUWTQxVWs&usp=sharing).

### Train on your own datasets

Alternatively, you can train the model on your own dataset. As preparations, you need:

1. Create a hdf5 file with size of NxCxHxW, where N is the total number of images, C is the number of channels, H is the height of image, and W the width of image. Then move it in datasets/*dataset_name*/data4torch.h5

2. Create a lua file to specify the network architecture for your dataset. Put it in models_def/*dataset_name*.lua.

3. Afterwards, you can run train.lua by specifying the dataset name as your own dataset. That's it!

### Compared Approaches

We upload the code for the compared approaches in matlab folder. Please refer to the original paper for details and cite them properly. In this foler, we also attach the evaluation code for two metric: normalized mutual information (NMI) and clustering accuracy (AC).

### Q&A

You are welcome to send message to (jw2yang at vt.edu) if you have any issue on this code.
