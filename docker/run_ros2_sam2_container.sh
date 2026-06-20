#!/bin/bash
# Run the ROS2 + SAM2 + FoundationPose container.
docker rm -f ros2_sam2_foundationpose
DIR=$(pwd)/../
xhost + && docker run --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  -it --network=host \
  --name ros2_sam2_foundationpose \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $DIR:$DIR -v /home:/home -v /mnt:/mnt \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp \
  --ipc=host \
  -e DISPLAY=${DISPLAY} \
  -e GIT_INDEX_FILE \
  ros2_sam2_foundationpose:latest bash -c "cd $DIR && bash"
  