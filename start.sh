#!/bin/bash
#debugging job
#SBATCH --job-name=debug_movie_review_job # specify the job name for monitoring
#SBATCH --output=transformer-out/moviereview_JOB_%j.out # specify the output file
#SBATCH --error=transformer-err/moviereview_JOB_%j.err # specify the error file
#SBATCH --nodes=1 # As we have single node it should be always set as 1
#SBATCH --cpus-per-task=4 # Number of CPUs
#SBATCH --gres=gpu:1g.10gb:1  # Allocate 1 GPU resources with specified configurations
#SBATCH --mem=50G  # Specify the total amount of memory
#SBATCH --time=72:00:00  # Set the time limit to 72 hours
#SBATCH --partition=debugging 
#SBATCH --qos=debugging
#SBATCH --account=debugging


# Run the Python script
srun hostname

# print MIG devices ids for debugging
echo $CUDA_VISIBLE_DEVICES

# Create a virtual environment and install the required packages
python -m venv ./debug-env
# Activate the virtual environment
source ./debug-env/bin/activate
# Install the required packages
bash install.sh

# Cd to the directory where the python script is located
python3  main.py