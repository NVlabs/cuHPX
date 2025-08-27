FROM nvcr.io/nvidia/pytorch:25.04-py3

# Update package lists and install dependencies
RUN apt-get update

COPY . /cuhpx
RUN cd /cuhpx && \
    pip install scikit-build-core>=0.8 && \
    pip install --no-build-isolation .

# Set the default command for the container
CMD ["/bin/bash"]
