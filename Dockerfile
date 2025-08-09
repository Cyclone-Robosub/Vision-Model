FROM ros:jazzy

# Install additional tools you might need
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /workspace