# This is used to test the speed of data augmentation on a single video
from src.data_augmentation import DataAugmentation

if __name__ == '__main__':
    performanceTest = False

    import sys, os
    from timeit import timeit
    from pprint import pprint
    
    dataset_path = "/test_videos"
    n = 10

    if performanceTest:
        times = {}
        for proc in range(1, 32 + 1):
            da = DataAugmentation(
                rotationsX=[-5],
                rotationsY=[-5],
                datasetFolder=dataset_path, 
                outputPath=f'{dataset_path}/augmentations',
                useGpu=False,
                numJobs=proc,
                disablePBar=True,
                deletePreviousAugs=True
            )
            result = timeit(stmt='da.createDataAugmentedVideos()', globals=globals(), number=n)
            times[proc] = result/n
        enablePrint()
        pprint(times)
    else: # We want to check if using OpenCV is different from Guru's implementation
        da = DataAugmentation(
            rotationsX=[-5],
            rotationsY=[-5],
            datasetFolder=dataset_path, 
            outputPath=f'{dataset_path}/augmentations',
            useGpu=False,
            numJobs=1,
            useOpenCVProjectPoints=True,
            disablePBar=False,
            deletePreviousAugs=False
        )
        da.createDataAugmentedVideos()

        da = DataAugmentation(
            rotationsX=[-5],
            rotationsY=[-5],
            datasetFolder=dataset_path, 
            outputPath=f'{dataset_path}/augmentations',
            useGpu=False,
            numJobs=1,
            useOpenCVProjectPoints=False,
            disablePBar=False,
            deletePreviousAugs=False
        )
        da.createDataAugmentedVideos()