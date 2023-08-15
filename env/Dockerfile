FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04

# Environment variables
ARG PYTHON_VERSION=3.8
ARG DEBIAN_FRONTEND=noninteractive

# NOTE: If you are in China, please uncomment the following lines.
# ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1
# RUN sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
#     sed -i "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list

# Install apt packages
RUN rm -f /etc/apt/sources.list.d/* &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends \
        # Ccommon tools
        autoconf automake autotools-dev build-essential ca-certificates \
        make cmake ninja-build pkg-config g++ ccache yasm \
        ccache doxygen graphviz plantuml \
        daemontools krb5-user ibverbs-providers libibverbs1 \
        libkrb5-dev librdmacm1 libssl-dev libtool \
        libnuma1 libnuma-dev libpmi2-0-dev \
        openssh-server openssh-client pkg-config nfs-common \
        ## Tools
        git curl wget unzip nano net-tools sudo htop iotop \
        cloc rsync xz-utils software-properties-common \
    && rm /etc/ssh/ssh_host_ecdsa_key \
    && rm /etc/ssh/ssh_host_ed25519_key \
    && rm /etc/ssh/ssh_host_rsa_key \
    && cp /etc/ssh/sshd_config /etc/ssh/sshd_config_bak \
    && sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
    && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
    && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Conda & Python
ENV PATH="/opt/conda/bin:${PATH}"
ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
RUN cd /tmp &&\
    curl --retry 3 -fsSL -O "https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh" &&\
    bash "./Miniconda3-py38_23.5.2-0-Linux-x86_64.sh" -b -p "/opt/conda" &&\
    rm -rf /tmp/*

RUN conda install python=${PYTHON_VERSION} &&\
    conda update --prefix "/opt/conda" --all -y &&\
    conda clean --all -f -y

# install openMPI
RUN mkdir /.openmpi/
RUN wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.3.tar.gz
RUN gunzip -c openmpi-4.0.3.tar.gz | tar xf - \
    && cd openmpi-4.0.3 \
    && ./configure --prefix=/.openmpi/ --with-cuda \
    && make all install

ENV PATH /.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH /.openmpi/lib:$LD_LIBRARY_PATH

# install pytorch, torchvision.
RUN git clone --recursive  https://github.com/pytorch/pytorch
RUN conda install -y numpy pyyaml &&\
    pip install typing-extensions
RUN cd pytorch && \
    git checkout tags/v1.8.0 && \
    git submodule sync && \
    git submodule update --init --recursive && \
    TORCH_CUDA_ARCH_LIST="3.7+PTX;5.0;6.0;6.1;7.0;7.5;8.0;8.6" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v .
RUN conda install -c conda-forge 'ffmpeg<5'
RUN git clone https://github.com/pytorch/vision.git && cd vision && git checkout tags/v0.9.2-rc2 && python setup.py install

# tmux
RUN rm -f /etc/apt/sources.list.d/* &&\
    apt-get update &&\
    apt-get -y install tzdata tmux &&\
    apt-get clean

# configure user
RUN groupadd --gid 65533 default_user
RUN useradd --shell /bin/bash \
    --create-home --home-dir /tmp/default_user \
    --uid 65533 --gid 65533 \
    default_user
RUN echo "default_user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN chown -R default_user:default_user /opt &&\
    chmod 755 /opt &&\
    chown -R default_user:default_user /var &&\
    chmod 755 /var

# extra requirements
RUN conda install scikit-learn pandas seaborn
RUN pip install --upgrade scikit-image &&\
    pip install wand einops opencv-python pyarrow lmdb

RUN apt -y install imagemagick
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt -y install gcc-9 libstdc++6

USER default_user
WORKDIR /home/default_user