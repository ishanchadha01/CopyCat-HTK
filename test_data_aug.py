from pickle import FALSE
import numpy as np
import cupy as cp
import os
import sys
import torch.multiprocessing as mp
import time

from pprint import pprint
from itertools import product
from functools import partial
from tqdm import tqdm  # Ensure that version is 4.51.0 to allow for nested progress bars
from p_tqdm import p_map
from src.data_augmentation import *
from cupyx.profiler import benchmark

CPU = False
sys.path.append(os.path.abspath('../'))

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # This is used to test the speed of data augmentation on a single video for CPU
    
    dataset_path = "/data/TestDataAug/test_video"
    
    if CPU:
        times = {}
        for proc in range(1, os.cpu_count() + 1):
            da = DataAugmentation(
                rotationsX=[-5],
                rotationsY=[-5],
                datasetFolder=dataset_path, 
                outputPath=f'{dataset_path}/augmentations',
                onlyGpu=False,
                numCpu=proc
            )
            start = time.perf_counter()
            da.createDataAugmentedVideos()
            end = time.perf_counter()
            times[proc] = end - start
        
        pprint(times)
    else: # Benchmark GPU
        video = np.load("/data/TestDataAug/test_video/test_video.npz")
        cameraIntrinsicMatrix = cp.array(video['cameraIntrinsicMatrix'])
        distortionCoefficients = cp.array(video['distortionCoefficients'])
        del video
        
        videoFrames = cp.load("/data/TestDataAug/test_video/test_video.npz")
        depthFrame = videoFrames['DepthFrame0']
        colorFrame = videoFrames['ColorFrame0']
        
        useBodyPixModel=1
        medianBlurKernelSize=5
        gaussianBlurKernelSize=55
        autoTranslate=True
        pointForAutoTranslate=(3840 // 2, 2160 // 2)
        
        gpu = True
        depthData = cleanDepthMap(depthFrame, colorFrame, useBodyPixModel,
                            medianBlurKernelSize, gaussianBlurKernelSize, gpu=gpu) / 1000
        pixels = createPixelCoordinateMatrix(depthData, gpu=gpu)
        rotation_x = getRotationMatrix(0, np.deg2rad(5), gpu=gpu)
        rotation_y = getRotationMatrix(1, np.deg2rad(0), gpu=gpu)
        worldGrid = calculateWorldCoordinates(pixels, cameraIntrinsicMatrix, gpu=gpu)
        translation = cp.array([0., 0., 0.])
        projectedImage = projectPoints(worldGrid, rotation_x.dot(rotation_y), translation, cameraIntrinsicMatrix, distortionCoefficients, gpu=gpu)

        if autoTranslate:
            Tx, Ty = solveForTxTy(
                pointForAutoTranslate, 5, 0, depthData, cameraIntrinsicMatrix)
            projectedImage[:, 0] += Tx
            projectedImage[:, 1] += Ty
        originalPixels = pixels[:, :-1]
        newImage = createNewImage(projectedImage, originalPixels, colorFrame, gpu=gpu)
        
        print(benchmark(cleanDepthMap, (depthFrame, colorFrame, useBodyPixModel, medianBlurKernelSize, gaussianBlurKernelSize, True), n_repeat=100, n_warmup=10))
        print()
        print(benchmark(createPixelCoordinateMatrix, (depthData, True), n_repeat=100, n_warmup=10))
        print()
        print(benchmark(getRotationMatrix, (0, np.deg2rad(5), True), n_repeat=100, n_warmup=10))
        print()
        print(benchmark(getRotationMatrix, (1, np.deg2rad(0), True), n_repeat=100, n_warmup=10))
        print()
        print(benchmark(calculateWorldCoordinates, (pixels, cameraIntrinsicMatrix, True), n_repeat=100, n_warmup=10))
        print()
        print(benchmark(projectPoints, (worldGrid, rotation_x.dot(rotation_y), translation, cameraIntrinsicMatrix, distortionCoefficients, True), n_repeat=100, n_warmup=10))
        print()
        print(benchmark(solveForTxTy, (pointForAutoTranslate, 5, 0, depthData, cameraIntrinsicMatrix), n_repeat=100, n_warmup=10))
        print()
        print(benchmark(createNewImage, (projectedImage, originalPixels, colorFrame, True), n_repeat=100, n_warmup=10))
        print()
        del depthData, pixels, rotation_x, rotation_y, worldGrid, translation, projectedImage, originalPixels, newImage
        print(benchmark(augmentFrame, (colorFrame, depthFrame, (5, 0), cameraIntrinsicMatrix, distortionCoefficients, useBodyPixModel, medianBlurKernelSize, gaussianBlurKernelSize, autoTranslate, pointForAutoTranslate, False, True), n_repeat=100, n_warmup=10))
        print()
        