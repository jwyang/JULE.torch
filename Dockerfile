FROM kaixhin/cuda-torch:7.5
MAINTAINER Sara Mousavi <mousavi@vols.utk.edu>

RUN luarocks install nn
RUN luarocks install image
RUN luarocks install cunn

WORKDIR /workspace/
RUN git clone https://github.com/Saulzar/lua-knn
RUN cd lua-knn/ && luarocks make
RUN git clone https://github.com/jwyang/JULE.torch
RUN apt-get install -y libhdf5-dev
RUN git clone https://github.com/anibali/torch-hdf5
RUN cd torch-hdf5/ && git checkout hdf5-1.10 && yes | luarocks make hdf5-0-0.rockspec

CMD /bin/bash
