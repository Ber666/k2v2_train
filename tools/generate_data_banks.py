import glob
import json
import sys

from tqdm import tqdm
from megatron.core.datasets.indexed_dataset import _IndexReader



# example run: python generate_data_banks.py /mbz/users/linghao.jin/data -new_250k_tokenized scaling_law
tokenized_data_path = sys.argv[1]
data_suffix=sys.argv[2]
data_bank_name = sys.argv[3]
save_path = "/mbz/users/linghao.jin/projects/LLM360-MoE/data_banks"

data_banks = {}

def generate(file, data):
    file_prefix = file.split('.idx')[0]
    dup = file_prefix.split('/')[-1]
    if dup.endswith("_text_document"):
        dup = dup[0:-14]
    print(f"{data}_{dup}")
    file_prefix = file_prefix.replace('/mbz/','/mnt/')
    index = _IndexReader(file, multimodal=False)
    seq_lens = index.sequence_lengths.tolist()
    doc_token_count = sum(seq_lens) 
    data_banks[f"{data}_{dup}"] = (doc_token_count, 1.0, file_prefix)
    
# all upsample
for folder in tqdm(glob.glob(f"{tokenized_data_path}/*{data_suffix}")):
    data = folder.split(data_suffix)[0].split("/")[-1]
    
    if data in ['DCLM-baseline', 'fineweb_1.5T', 'SlimPajama-627B', 'proof-pile-2']:
        continue
    if data != "common-crawl":            
        for file in glob.glob(f"{tokenized_data_path}/{data}{data_suffix}/*.idx"):
            if data.startswith("the-stack-v2-train-full-ids"):
                data = "stackV2"
            generate(file, data)
    else: 
        for file in glob.glob(f"{tokenized_data_path}/{data}{data_suffix}/merged/*.idx"):
            generate(file, "cc")
    

# certain file
# for i, file in enumerate(tqdm(glob.glob(f"{tokenized_data_path}/*.idx"))):
#     file_prefix = file.split('.idx')[0]
#     file_prefix = file_prefix.replace('/mbz/','/mnt/')
#     index = _IndexReader(file, multimodal=False)
#     seq_lens = index.sequence_lengths.tolist()
#     doc_token_count = sum(seq_lens) 
#     data_banks[f'prefix_{i}'] = (doc_token_count, 1.0, file_prefix)

with open(f"{save_path}/{data_bank_name}_databanks.json", "w") as f:
    json.dump(data_banks, f)
