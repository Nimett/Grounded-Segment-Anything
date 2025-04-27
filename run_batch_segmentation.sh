#! /bin/bash

# Check if the Docker image gsa:v0 exists
if ! docker images | grep -q "gsa\s*v0"; then
    echo "Docker image gsa:v0 not found. Building the image..."
    make build-image
else
    echo "Docker image gsa:v0 found."
fi
# make build-image

parent_output_dir=$1
bag_file_name=$2
image_extension=$3
seg_classes=$4

image_extension=${image_extension:-"png"}
seg_classes=${seg_classes:-"High-standing platforms,Ground,Humans"}

cmd="parent_output_dir=$parent_output_dir && \
bag_file_name=$bag_file_name && \
image_extension=$image_extension && \
seg_classes=\"$seg_classes\" && "

cmd+="export HF_HOME=/tmp  && \
    cd Grounded-Segment-Anything/ && \
    exec python segment_images_batch.py \
    --parent_output_dir $parent_output_dir \
    --bag_file_name $bag_file_name \
    --image_extension $image_extension \
    --seg_classes \"$seg_classes\""

# cmd+="export HF_HOME=/tmp  && \
#     python -c \"import torch; print(torch.cuda.is_available())\""

echo $cmd

exec docker run \
    --gpus 1 \
    -v $parent_output_dir:$parent_output_dir \
    --user $(id -u):$(id -g) \
    gsa:v0 \
    /bin/bash -c "$cmd"