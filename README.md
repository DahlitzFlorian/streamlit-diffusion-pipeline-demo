# Streamlit Diffusion Pipeline Demo on IBM POWER

## Description

This little demo shows how to use a diffusion pipeline in conjunction with streamlit on IBM POWER (ppc64le).

## Usage

Clone the repository and build the image:

```shell
$ podman image build -t streamlit-diffusion .
```

... and start the container with a volume for the models, a mapped port, and passed GPUs:

```shell
$ podman container run --name streamlit-diffusion --rm -itd --device nvidia.com/gpu=all -v huggingface_cache:/root/.cache/huggingface/hub -p 8501:8501 streamlit-diffusion
```

Alternatively, use the `start.sh` script.
