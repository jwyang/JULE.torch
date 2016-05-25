# Joint Unsupervised Learning of Deep Representations and Image Clusters.

### Overview

This project is a Torch implementation for our CVPR 2016 paper, which performs jointly unsupervised learning of deep CNN and image clusters. The intuition behind this is that better image representation will facilitate clustering, while better clustering results will help representation learning. Given a unlabeled dataset, it will iteratively learn CNN parameters unsupervisedly and cluster images.

### Disclaimer
This is a torch version reimplementation to the code used in our CVPR paper. There is a slight difference between the code used to report the results in our paper. We will release the original Caffe version used in our CVPR 2016 paper as well.

### License

This code is released under the MIT License (refer to the LICENSE file for details).

### Citation
If you find our code is useful in your researches, please consider citing:

    @inproceedings{yangCVPR2016joint,
        Author = {Jianwei Yang and Devi Parikh and Dhruv Batra},
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

2. [lua-knn](https://github.com/Saulzar/lua-knn). It is used to compute the distance between neighbor samples. Go into the folder, and then compile the code as follows:

   ```bash
   $ luarocks install knn-0-1.rockspec 
   ```

### Run code


