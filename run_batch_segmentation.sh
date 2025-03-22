#! /bin/bash

# Check if the Docker image gsa:v0 exists
if ! docker images | grep -q "gsa\s*v0"; then
    echo "Docker image gsa:v0 not found. Building the image..."
    make build-image
else
    echo "Docker image gsa:v0 found."
fi
# make run

image_folder=$1
output_folder=$2
image_extension=$3
seg_classes=$4

image_extension=${image_extension:-"png"}
seg_classes=${seg_classes:-"High-standing platforms, Ground, Humans"}

cmd="image_folder=$image_folder && \
output_folder=$output_folder && \
image_extension=$image_extension && \
seg_classes=\"$seg_classes\" && "

cmd+="export HF_HOME=/tmp  && \
    cd Grounded-Segment-Anything/ && \
    exec python segment_images_batch.py \
    --image_folder $image_folder \
    --output_folder $output_folder \
    --image_extension $image_extension \
    --seg_classes \"$seg_classes\""

# cmd+="export HF_HOME=/tmp  && \
#     python -c \"import torch; print(torch.cuda.is_available())\""

echo $cmd

exec docker run \
    --gpus 1 \
    -v $image_folder:$image_folder \
    -v $output_folder:$output_folder \
    --user $(id -u):$(id -g) \
    gsa:v0 \
    /bin/bash -c "$cmd"