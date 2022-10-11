import os
import numpy as np

def create_batches(videos, num_batches, output_path):
    if not os.path.exists(f'{output_path}/batch_{num_batches}.txt'):
        batches = np.array_split(np.array(videos), num_batches)
        for i, batch in enumerate(batches):
            with open(f'{output_path}/batch_{i+1}.txt', 'w') as f:
                for video in batch:
                    f.write(f'{video} + \n')