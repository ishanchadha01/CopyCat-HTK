import os
import numpy as np
import shutil

def create_batches(videos, num_batches, output_path):
    if os.path.exists(f'{output_path}/batch_{num_batches}.txt') and not os.path.exists(f'{output_path}/batch_{num_batches+1}.txt'):
        print(f"{num_batches} batches already exists inside {output_path}")
        return
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    batches = np.array_split(np.array(videos), num_batches)
    for i, batch in enumerate(batches):
        with open(f'{output_path}/batch_{i+1}.txt', 'w') as f:
            for video in batch: 
                f.write(f'{video}\n')