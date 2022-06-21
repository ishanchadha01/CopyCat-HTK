# Imports
import cv2
import numpy as np
import os
import sys

from pyk4a import PyK4APlayback
from itertools import product
from functools import partial
from tqdm import tqdm  # Ensure that version is 4.51.0 to allow for nested progress bars
from p_tqdm import p_map
from .data_augmentation_utils import *
from src.openpose_feature_extraction.generate_mediapipe_features import extract_mediapipe_features

# Adds the src folder to the path so generate_mediapipe_features.py can be imported
sys.path.append(os.path.abspath('../'))


class DataAugmentation():
    """DataAugmentation is a class that contains all the data augmentation methods and performs data augmentation to create new videos"""

    def __init__(self, rotationsX, rotationsY, datasetFolder='./CopyCatDatasetWIP', outputPath='.', numJobs=os.cpu_count(), useBodyPixModel=1, medianBlurKernelSize=5, gaussianBlurKernelSize=55, autoTranslate=True, pointForAutoTranslate=(3840 // 2, 2160 // 2)):
        """__init__ initialized the Data Augmentation object with the required parameters

        Arguments:
            rotationsX {list} -- list of rotations in the x-axis
            rotationsY {list} -- list of rotations in the y-axis

        Keyword Arguments:
            datasetFolder {str} -- the path where the original videos are located (default: {'./CopyCatDatasetWIP'})
            outputPath {str} -- the path where the augmented videos will be saved (default: {'.'})
            useBodyPixModel {int} -- TensorFlow's bodypix model used to create better depth maps. Look at dictionary above for more information (default: {1})
            medianBlurKernelSize {int} --  kernel size for the median blur filter applied to depth maps (default: {5})
            gaussianBlurKernelSize {int} -- kernel size for the gaussian blur filter applied to depth maps (default: {55})
            autoTranslate {bool} -- whether to auto translate the picture so that an initial point remains in the same place (default: {True})
            pointForAutoTranslate {tuple} -- the point about which auto translation is calculated (default: {(3840 // 2, 2160 //2)})

        Raises:
            NameError: If the dataset folder does not exist
            ValueError: If the X rotation is greater than 90 or less than 0
            ValueError: If the Y rotation is greater than 90 or less than 0
            TypeError: The length of the point for auto translate is not 2
            ValueError: The point for auto translate is not within the image
        """
        # If the dataset folder or the output folder does not exist, raise an error
        if not os.path.exists(datasetFolder):
            raise NameError(f'Dataset folder {datasetFolder} does not exist')

        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
            print("Output path does not exist. Creating output path...")
            os.makedirs(outputPath)
    
        # Delete the previous augmentations before making new ones
        if os.path.exists(outputPath):
            os.system(f'rm -rf {outputPath}')
        os.makedirs(outputPath)

        # If the x or y angle is negative or greater than 90, raise an error
        for x, y in zip(rotationsX, rotationsY):
            if abs(x) >= 90:
                raise ValueError('X Rotation must be less than 90')
            elif abs(y) >= 90:
                raise ValueError('Y Rotation must be less than 90')

        if useBodyPixModel not in bodyPixModelsDict:
            raise ValueError(
                f'useBodyPixModel must be one of {list(bodyPixModelsDict.keys())}')

        if autoTranslate and type(pointForAutoTranslate) != 'tuple' and len(pointForAutoTranslate) != 2:
            raise TypeError(
                'Point for auto translate must be a tuple of length 2')

        if pointForAutoTranslate[0] > 3840 or pointForAutoTranslate[1] > 2160 or pointForAutoTranslate[0] < 0 or pointForAutoTranslate[1] < 0:
            raise ValueError(
                'Point for auto translate must be within the bounds of the image')

        # Initializing Mediapipe (so you don't get repeating log messages saying "INFO: Created TensorFlow Lite XNNPACK delegate for CPU.")
        extract_mediapipe_features(frames=None, save_filepath=None)

        # [combination for combination in list(product(rotationsX, rotationsY)) if combination != (0, 0)]
        self.rotations = list(product(rotationsX, rotationsY))
        self.datasetFolder = datasetFolder
        self.numJobs = numJobs
        self.medianBlurKernelSize = medianBlurKernelSize
        self.gaussianBlurKernelSize = gaussianBlurKernelSize
        self.useBodyPixModel = useBodyPixModel
        self.outputPath = outputPath
        self.autoTranslate = autoTranslate
        self.pointForAutoTranslate = pointForAutoTranslate

        # Get the list of videos
        self.listOfVideos = getListVideos(self.datasetFolder)

        # min_v_0, min_v_2160, min_u_0, min_u_3840 = self.calculateMinRotationsPossible()

    def __str__(self) -> str:
        """__str__ returns a string representation of the DataAugmentation Object when used in print statements

        Returns:
            str -- string representation of the DataAugmentation Object (when used in print statements)
        """
        rotationStr = "Data Augmentation With Rotations (X, Y):\n"
        for rotation in self.rotations:
            rotationStr += f"{str(rotation)}\n"

        return rotationStr

    def __repr__(self) -> str:
        """__repr__ returns a string representation of the DataAugmentation Object when used in a list/dict/etc.

        Returns:
            str -- returns a string representation of the DataAugmentation Object when used in a list/dict/etc.
        """
        return f"DataAugmentation(rotations={self.rotations})"

    def createDataAugmentedVideos(self) -> list:
        """createDataAugmentedVideos is the main function. It creates the data augmented videos. 

        *** SHOULD BE THE ONLY METHOD CALLED FROM THIS CLASS ***

        Returns:
            list -- list of output paths of all the augmented videos
        """
        # Get the list of all the videos within the dataset folder
        self.pbarAllVideos = tqdm(total=len(self.listOfVideos))
        self.pbarAllVideos.set_description("Total Videos Done")
        self.pbarAllVideosRotations = tqdm(
            total=len(self.listOfVideos) * len(self.rotations))
        self.pbarAllVideosRotations.set_description(
            "Total Combinations Done")
        # Start applying data augmentation to each video while appending the augmented video path to a new list
        newJSONs = []
        for video in self.listOfVideos:
            for rotation in self.rotations:
                newJSONs.append(self.augmentVideo(video, rotation=rotation))
                self.pbarAllVideosRotations.update(1)
            self.pbarAllVideos.update(1)

        return newJSONs

    def augmentVideo(self, video, rotation) -> str:
        """augmentVideo augments a video with a given rotation

        Arguments:
            video {str} -- the path of the video to augment
            rotation {list} -- list of tuple containing X and Y rotation to apply to the video

        Returns:
            str -- destination path of the augmented video
        """
        # Get the video name by getting the last str when splitting on '/'
        videoName = video.split('/')[-1][:-4]
        user = video.split('/')[4].split('_')[1]

        # Extract the camera calibrations used in cv2.projectPoints
        intrinsicCameraMatrix = getCameraIntrinsicMatrix(video)
        distortionCoefficients = getDistortionCoefficients(video)

        rotationName = f"rotation({rotation[0]},{rotation[1]})"
        fullRotationName = f"{rotationName}-autoTranslate({self.autoTranslate})-pointForAutoTranslate({self.pointForAutoTranslate[0]},{self.pointForAutoTranslate[1]})"
        currJSONPath = f'{self.outputPath}/{fullRotationName}-{user}-{videoName}.json'

        # If the augmented video exists, then there's no need to run data augmentation again. Only do this if the augmented video does not exist
        videoFrames = np.load(f"{video}.npz")
        if videoFrames['DepthFrame0'].shape[0] < 2160:
            return None
        allImages = [videoFrames[image] for image in getColorFrames(video)]
        allDepth = [videoFrames[depth] for depth in getDepthFrames(video)]
        del videoFrames # Clearing variable to decrease RAM usage

        # Parallelize the frame augmentation process to speed up the process
        augmentedFrames = p_map(
            partial(
                augmentFrame,
                rotation=rotation, 
                cameraIntrinsicMatrix=intrinsicCameraMatrix, 
                distortionCoefficients=distortionCoefficients,
                useBodyPixModel=self.useBodyPixModel, 
                medianBlurKernelSize=self.medianBlurKernelSize, 
                gaussianBlurKernelSize=self.gaussianBlurKernelSize, 
                autoTranslate=self.autoTranslate, 
                pointForAutoTranslate=self.pointForAutoTranslate
            ),
            allImages, 
            allDepth,
            num_cpus=self.numJobs,
            desc=f"{user}-{rotationName}"
        )
        
        # Clearing variables to decrease RAM usage
        del allImages
        del allDepth

        # Extract the mediapipe features for every frame
        extract_mediapipe_features(augmentedFrames, currJSONPath)
        
        return currJSONPath

    def calculateMinRotationsPossible(self) -> tuple:
        """calculateMinRotationsPossible returns the minimum rotations possible in the 2 main axes (going up and down)
        For more information on how the math works out, check the Sign-Recognition channel or ask Guru

        Returns:
            tuple -- the minimum rotation in four directions returned in this order: left, right, up, down
        """
        # Get the 3D mediapipe extractions for each video and flatten poseFeatures so it's just a big Nx3 numpy array
        poseFeatures = p_map(get3DMediapipeCoordinates, self.listOfVideos[:2], num_cpus=self.numJobs, desc="Getting 3D mediapipe features")
        cameraIntrinsicMatrices = p_map(getCameraIntrinsicMatrix, self.listOfVideos[:2], num_cpus=self.numJobs, desc="Getting camera matrices")

        # Initial minimum values set to infinity
        min_v_0 = np.inf
        min_v_2160 = np.inf
        min_u_0 = np.inf
        min_u_3840 = np.inf

        pbarCalculateMinRotations = tqdm(total=len(poseFeatures) * 4, desc="Calculating min rotations")

        # Apply all the 4 rotations possible to each point by iterating over each video
        for videoPoseFeature, cameraIntrinsicMatrix in zip(poseFeatures, cameraIntrinsicMatrices):
            pbarCalculateMinRotations.set_description("Rotation V=0")
            curr_rotations_v_0 = np.apply_along_axis(lambda row: rotation_v_0(
                row[0], row[1], row[2], cameraIntrinsicMatrix), 1, videoPoseFeature)
            pbarCalculateMinRotations.update(1)
            
            pbarCalculateMinRotations.set_description("Rotation V=2160")
            curr_rotations_v_2160 = np.apply_along_axis(lambda row: rotation_v_2160(
                row[0], row[1], row[2], cameraIntrinsicMatrix), 1, videoPoseFeature)
            pbarCalculateMinRotations.update(1)
            
            pbarCalculateMinRotations.set_description("Rotation U=0")
            curr_rotations_u_0 = np.apply_along_axis(lambda row: rotation_u_0(
                row[0], row[1], row[2], cameraIntrinsicMatrix), 1, videoPoseFeature)
            pbarCalculateMinRotations.update(1)
            
            pbarCalculateMinRotations.set_description("Rotation U=3840")
            curr_rotations_u_3840 = np.apply_along_axis(lambda row: rotation_u_3840(
                row[0], row[1], row[2], cameraIntrinsicMatrix), 1, videoPoseFeature)
            pbarCalculateMinRotations.update(1)
            
            # Calculate the minimum values and replace the minimum values if they are smaller
            curr_min_v_0 = np.min(curr_rotations_v_0)
            curr_min_v_2160 = np.min(curr_rotations_v_2160)
            curr_min_u_0 = np.min(curr_rotations_u_0)
            curr_min_u_3840 = np.min(curr_rotations_u_3840)
            if curr_min_v_0 < min_v_0:
                min_v_0 = curr_min_v_0
            if curr_min_v_2160 < min_v_2160:
                min_v_2160 = curr_min_v_2160
            if curr_min_u_0 < min_u_0:
                min_u_0 = curr_min_u_0
            if curr_min_u_3840 < min_u_3840:
                min_u_3840 = curr_min_u_3840

        # Return these minimum rotations
        return min_v_0, min_v_2160, min_u_0, min_u_3840
