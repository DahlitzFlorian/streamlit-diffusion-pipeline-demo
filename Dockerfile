FROM almalinux:8

ARG PYTHON_VERSION=3.9
ARG CUDA_VERSION=11.8.0

USER root

# set prefix
ENV CONDA_DIR=/opt/conda
ENV MAMBA_ROOT_PREFIX=${CONDA_DIR}
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# create conda directory
RUN mkdir -p "${CONDA_DIR}"

RUN dnf -y groupinstall "Development Tools"

# epel
RUN dnf -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(rpm -E %rhel).noarch.rpm

# openblas
RUN dnf config-manager --enable powertools

RUN dnf install \
    -y \
    bzip2 \
    openblas-devel \
    wget

WORKDIR /root

# Install micromamba
RUN wget -qO /tmp/micromamba.tar.bz2 \
        "https://micromamba.snakepit.net/api/micromamba/linux-ppc64le/latest" && \
    tar -xvjf /tmp/micromamba.tar.bz2 --strip-components=1 bin/micromamba && \
    rm /tmp/micromamba.tar.bz2

# Install CUDA
RUN ./micromamba install \
    --yes \
    --root-prefix="${CONDA_DIR}" \
    --prefix="${CONDA_DIR}" \
    -c nvidia \
    cuda

# Install dependencies
RUN ./micromamba install \
    --yes \
    --root-prefix="${CONDA_DIR}" \
    --prefix="${CONDA_DIR}" \
    python=${PYTHON_VERSION} \
    "anaconda::cmake>=3.26" \
    anaconda::pillow \
    anaconda::pip \
    # needs to be the latest available version \
    "conda-forge::pyarrow>=11" \
    rocketce::pytorch-base

RUN pip install --upgrade pip

RUN pip install --prefer-binary \
    diffusers \
    transformers \
    accelerate \
    safetensors

RUN pip install \
    streamlit

RUN mkdir -p /root/.streamlit

RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

COPY t.py t.py

EXPOSE 8501

ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:4096"

CMD ["streamlit", "run", "t.py", "--server.port=8501", "--server.address=0.0.0.0"]
