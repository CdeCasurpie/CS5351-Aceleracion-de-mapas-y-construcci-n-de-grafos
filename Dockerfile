FROM ubuntu:jammy

ARG DEBIAN_FRONTEND=noninteractive
ARG uid
ARG gid
ARG user
ARG group

# Basic system update and timezone setup
RUN apt-get update
RUN apt-get install -y tzdata
RUN apt-get upgrade -y

# Add useful aliases
RUN echo 'alias ll="ls -l --color -a"' >> /root/.bashrc

# Basic packages + compilation + Python
RUN apt-get install -y python3 python3-pip python3-dev
RUN apt-get install -y build-essential cmake git pkg-config
RUN apt-get install -y pybind11-dev

# Install all CGAL dependencies
RUN apt-get install -y xz-utils
RUN apt-get install -y g++
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libgmp-dev
RUN apt-get install -y libmpfr-dev

# Install CGAL from system package
RUN apt-get install -y libcgal-dev

# Install CGAL from source (if specific version needed)
RUN mkdir -p /tmp/cgal
COPY CGAL-6.0.1.tar.xz /tmp/cgal/
RUN cd /tmp/cgal && tar -xf CGAL-6.0.1.tar.xz
RUN cd /tmp/cgal/CGAL-6.0.1 && mkdir -p build
RUN cd /tmp/cgal/CGAL-6.0.1/build && cmake ..
RUN cd /tmp/cgal/CGAL-6.0.1/build && make install
RUN rm -rf /tmp/cgal

# Clean up apt cache
RUN rm -rf /var/lib/apt/lists/*

# Create workspace directory with proper permissions BEFORE creating user
RUN mkdir -p /workspace

# Create non-root user
RUN groupadd -g ${gid} ${group} || true
RUN useradd -m -u ${uid} -g ${gid} -s /bin/bash ${user} || true

# Add aliases for user
RUN echo 'alias ll="ls -l --color -a"' >> /home/${user}/.bashrc || true

# Set ownership of workspace to user
RUN chown -R ${uid}:${gid} /workspace

# Now switch to user
USER ${user}
WORKDIR /workspace

# Copy source code with proper ownership
COPY --chown=${user}:${group} . /workspace/

# Install Python requirements
RUN pip3 install -r requirements.txt

# Compile matcher automatically
RUN mkdir -p build
WORKDIR /workspace/build
RUN cmake .. && make -j$(nproc)

WORKDIR /workspace
