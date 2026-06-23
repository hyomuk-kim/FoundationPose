#!/bin/bash
# Run the ROS2 + SAM2 + FoundationPose container.

# Notes on the flags that were required during bring-up:
#   --privileged + -v /dev:/dev : RealSense USB access from inside the container
#                                 (otherwise RS2_USB_STATUS_NO_DEVICE).
#   --network=host              : DDS auto-discovery across machines; match
#                                 ROS_DOMAIN_ID on both ends.
#   PYTHONPATH=''               : start each shell clean; the ros2 env's ROS2
#                                 setup otherwise leaks its 3.11 site-packages
#                                 into the 3.9 `my` env and breaks numpy.

docker rm -f ros2_sam2_foundationpose
DIR=$(pwd)/../
xhost + && docker run --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  -it --network=host \
  --name ros2_sam2_foundationpose \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --privileged \
  -v /dev:/dev \
  -v $DIR:$DIR -v /home:/home -v /mnt:/mnt \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp \
  --ipc=host \
  -e DISPLAY=${DISPLAY} \
  -e GIT_INDEX_FILE \
  ros2_sam2_foundationpose:latest bash -c "cd $DIR && export PYTHONPATH='' && bash"
