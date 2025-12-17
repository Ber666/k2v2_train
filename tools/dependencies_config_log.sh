# this is for the FM cluster

module load cuda/12.1
eval "$(/lustre/apps/apps/anaconda3/anaconda3-2023.03/bin/conda shell.bash hook)"

conda create -n llm360 python=3.9
conda activate llm360

# for cuda12-1
pip3 install torch

# on a GPU node (do not do on login node)
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

