#FROM nvidia/cuda:12.3.0-devel-ubuntu22.04
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Update package lists and install dependencies
RUN apt-get update

COPY . /cuhpx
RUN cd /cuhpx && \
    python setup.py build && \
    pip install .

# Set the default command for the container
CMD ["/bin/bash"]
