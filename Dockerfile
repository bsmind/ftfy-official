# Dockerfile for building ftfy-official image
FROM nvidia/cuda:9.0-base-ubuntu16.04 as base

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	cuda-command-line-tools-9-0 \
	cuda-cublas-9-0 \
	cuda-cufft-9-0 \
	cuda-curand-9-0 \
	cuda-cusolver-9-0 \
	cuda-cusparse-9-0 \
	libcudnn7=7.4.2.24-1+cuda9.0 \
	libfreetype6-dev \
	libhdf5-serial-dev \
	libpng12-dev \
	libzmq3-dev \
	pkg-config \
	software-properties-common \
	unzip \
	&& \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# In the Ubuntu 16.04 imaes, cudnn is placed in system paths. Move them
# to /usr/local/cuda
#RUN cp -P /usr/include/cudnn.h /usr/local/cuda/include
#RUN cp -P /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64

# Installs TensorRT, which is not included in NVIDIA Docker containers
RUN apt-get update \
	&& apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda9.0 \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda9.0 \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

# For CUDA profiling, TensorFlow requires CUPTI
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# install other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	curl \
	python \
	python3 \
	python3-dev \
	python-pip \
	python3-pip \
	python3-setuptools \
	tmux \
	vim \
	git \
	htop \
	&& \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/* 

RUN pip3 install --upgrade pip setuptools

# install application dependencies
RUN pip3 --no-cache-dir install \
	ipykernel \
	jupyter \
	Pillow \
	h5py \
	matplotlib \
	mock \
	numpy \
	scipy \
	sklearn \
	scikit-image \
	tqdm \
	pandas 

# install TF specific version
RUN pip3 install \
	tensorflow==1.12.0 \
	tensorflow-gpu==1.12.0

# copy ftfy code
#RUN git clone -b v100 https://github.com/bsmind/ftfy-official.git /ftfy-official/models
RUN mkdir -p /ftfy


# jupyter notebook (password: root)
WORKDIR /ftfy
#RUN pip3 install jupyter matplotlib
#RUN pip3 install jupyter_http_over_ws
#RUN jupyter serverextension enable --py jupyter_http_over_ws
RUN jupyter notebook -y --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py


# Tensorboard
EXPOSE 6006
# IPython
EXPOSE 8888

# run main process
#RUN ${PYTHON} -m ipykernel.kernelspec
RUN python3 -m ipykernel.kernelspec
CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/ftfy", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
