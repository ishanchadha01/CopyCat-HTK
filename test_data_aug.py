# This is used to test the speed of data augmentation on a single video
from src.data_augmentation import DataAugmentation
if __name__ == '__main__':
    import sys, os
    from timeit import timeit
    from pprint import pprint

    # Disable
    def blockPrint():
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint():
        sys.stdout = sys.__stdout__
    
    dataset_path = "/test_videos"
    n = 10

    times = {}
    blockPrint()
    for proc in range(1, 32 + 1):
        da = DataAugmentation(
            rotationsX=[-5],
            rotationsY=[-5],
            datasetFolder=dataset_path, 
            outputPath=f'{dataset_path}/augmentations',
            useGpu=False,
            numJobs=proc
        )
        result = timeit(stmt='da.createDataAugmentedVideos()', globals=globals(), number=n)
        times[proc] = result/n
    enablePrint()
    pprint(times)