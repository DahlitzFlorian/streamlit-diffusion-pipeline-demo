podman image build -t streamlit-diffusion .
podman container run --name streamlit-diffusion --rm -itd --device nvidia.com/gpu=all -v huggingface_cache:/root/.cache/huggingface/hub -p 8501:8501 streamlit-diffusion
