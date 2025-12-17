import os
import sys
import glob
import re


CKPT_DIR="/mbz/users/linghao.jin/ckpts"
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python delete_checkpoints.py ckpt_name")
        sys.exit(1)
    
    ckpt_path = os.path.join(CKPT_DIR, sys.argv[1])
    
    if not os.path.exists(ckpt_path):
        print("Checkpoint path does not exist")
        sys.exit(1)
    
    # keep last 5 checkpoints 
    checkpoints = glob.glob(f"{ckpt_path}/iter_*")
    ckpt_nums = [re.findall(r'\d+', ckpt)[-1] for ckpt in checkpoints]
    ckpt_nums = sorted(ckpt_nums, key=lambda x: int(x))
    
    for ckpt_num in ckpt_nums[:-5]:
        if int(ckpt_num) % 5000 == 0:  # 5000x checkpoints only delete optimizer files
            print(f"Deleting optimizer files for checkpoint {ckpt_num}")
            optimizer_files = glob.glob(f"{ckpt_path}/iter_{ckpt_num}/*/distrib_optim.pt")
            for of in optimizer_files:
                os.remove(of)
        
        else: # others delete whole folder.
            ckpt_files = glob.glob(f"{ckpt_path}/iter_{ckpt_num}/*")
            print(f"Deleting checkpoint {ckpt_num}")
            for cf in ckpt_files:
                for _ in glob.glob(f"{cf}/*"):
                    os.remove(_)
                os.rmdir(cf)
            os.rmdir(f"{ckpt_path}/iter_{ckpt_num}")
            