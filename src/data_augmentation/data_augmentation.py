# Imports
import cv2
import numpy as np
import os

from pyk4a import PyK4APlayback
from pytransform3d.rotations import active_matrix_from_angle
from itertools import product
from tqdm import tqdm # Ensure that version is 4.51.0 to allow for nested progress bars
from data_augmentation_utils import *

class DataAugmentation():
    """DataAugmentation is a class that contains all the data augmentation methods and performs data augmentation to create new videos"""
    def __init__(self, rotationsX, rotationsY, datasetFolder='./CopyCatDatasetWIP', outputPath='.', useBodyPixModel=1, medianBlurKernelSize=5, gaussianBlurKernelSize=55, autoTranslate=True, pointForAutoTranslate=(3840 // 2, 2160 //2)):
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
            NameError: If the output path does not exist
            ValueError: If the X rotation is greater than 90 or less than 0
            ValueError: If the Y rotation is greater than 90 or less than 0
            TypeError: The length of the point for auto translate is not 2
            ValueError: The point for auto translate is not within the image
        """
        # If the dataset folder or the output folder does not exist, raise an error
        if not os.path.exists(datasetFolder):
            raise NameError(f'Dataset folder {datasetFolder} does not exist')
        
        if not os.path.exists(outputPath):
            raise NameError(f'Output path {outputPath} does not exist')
                
        # If the x or y angle is negative or greater than 90, raise an error
        for x, y in zip(rotationsX, rotationsY):
            if abs(x) >= 90:
                raise ValueError('X Rotation must be less than 90')
            elif abs(y) >= 90:
                raise ValueError('Y Rotation must be less than 90')
        
        if useBodyPixModel not in bodyPixModelsDict:
            raise ValueError(f'useBodyPixModel must be one of {bodyPixModelsDict.keys()}')
            
        if autoTranslate and type(pointForAutoTranslate) != 'tuple' and len(pointForAutoTranslate) != 2:
            raise TypeError('Point for auto translate must be a tuple of length 2')
        
        if pointForAutoTranslate[0] > 3840 or pointForAutoTranslate[1] > 2160 or pointForAutoTranslate[0] < 0 or pointForAutoTranslate[1] < 0:
            raise ValueError('Point for auto translate must be within the bounds of the image')
            
        self.rotations = list(product(rotationsX, rotationsY)) #[combination for combination in list(product(rotationsX, rotationsY)) if combination != (0, 0)]
        self.datasetFolder = datasetFolder
        self.medianBlurKernelSize = medianBlurKernelSize
        self.gaussianBlurKernelSize = gaussianBlurKernelSize
        self.useBodyPixModel = useBodyPixModel
        self.outputPath = outputPath
        self.autoTranslate = autoTranslate
        self.pointForAutoTranslate = pointForAutoTranslate
    
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
        listOfVideos = getListVideos(self.datasetFolder)
        self.pbarAllVideos = tqdm(total=len(listOfVideos))
        self.pbarAllVideos.set_description("Total Videos Done")
        self.pbarAllVideosRotations = tqdm(total=len(listOfVideos) * len(self.rotations))
        self.pbarAllVideosRotations.set_description("Total Videos Done For 1 Combination")
        # Start applying data augmentation to each video while appending the augmented video path to a new list
        newVideos = []
        for video in listOfVideos:
            for rotation in self.rotations:
                newVideos.append(self.augmentVideo(video, rotation=rotation))
                self.pbarAllVideosRotations.update(1)
            self.pbarAllVideos.update(1)
               
        return newVideos
    
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
        user = videoName.split('_')[1]
        
        # Open the videos. Using OpenCV to get the RGB values and using PyK4A to get the depth maps.
        playbackDepth = PyK4APlayback(video)
        playbackDepth.open()
        playbackImage = cv2.VideoCapture(video)

        # Extract the camera calibrations used in cv2.projectPoints
        intrinsicCameraMatrix = playbackDepth.calibration.get_camera_matrix(camera=1)
        distortionCoefficients = playbackDepth.calibration.get_distortion_coefficients(camera=1)

        # Define new video
        codec = cv2.VideoWriter_fourcc('M','J','P','G')
        fps = playbackImage.get(cv2.CAP_PROP_FPS)
        
        rotationName = f"rotation({rotation[0]},{rotation[1]})"
        
        currVideoPath = f'./{self.outputPath}/{rotationName}_{videoName}.avi'
        frameWidth = int(playbackImage.get(3))
        frameHeight = int(playbackImage.get(4))
        
        # If the augmented video exists, then there's no need to run data augmentation again. Only do this if the augmented video does not exist
        if not os.path.exists(currVideoPath):
            out = cv2.VideoWriter(currVideoPath, codec, fps, (frameWidth, frameHeight))
            frameCount = countFrames(video)
            pbarFrame = tqdm(total=frameCount)
            pbarFrame.set_description(f"{user}-{rotationName}")
            
            # Iterate over the video and get the projected image. Then write that image to the new video
            while (True):
                try:
                    
                    capture = playbackDepth.get_next_capture()
                    _, image = playbackImage.read()
                    
                    newImage = self.augmentFrame(image, capture, rotation, intrinsicCameraMatrix, distortionCoefficients)

                    # For some reason, I'm not needing to convert from BGR to RGB. It's already RGB                
                    out.write(newImage)
                    pbarFrame.update(1)
                    
                except EOFError:
                    break
            
            # Close all the videos
            playbackDepth.close()
            out.release()
            playbackImage.release()
            
        return currVideoPath
    
    def augmentFrame(self, image, capture, rotation, cameraIntrinsicMatrix, distortionCoeffients) -> np.ndarray:
        """augmentFrameRotation rotates the current frame by the given rotation
        Arguments:
            image {np.ndarray} -- RGB image of the current frame
            capture {pyk4a.capture} -- A capture object containing the depth data for the current frame
            rotation {list} -- list of tuple containing X and Y rotation to apply to the video
            cameraIntrinsicMatrix {np.ndarray} -- 3x3 matrix explaining focal length, principal point, and aspect ratio of the camera
            distortionCoeffients {[type]} -- 1x8 matrix explaining the distortion of the camera

        Returns:
            np.ndarray -- RGB projected image of the current frame given the rotation
        """

        # Clean the depth map and divide by 1000 to convert millimeters to meters
        depth = cleanDepthMap(capture.transformed_depth, image, self.useBodyPixModel, self.medianBlurKernelSize, self.gaussianBlurKernelSize)
        depthData = depth / 1000
        
        # Define a matrix that contains all the pixel coordinates and their depths in a 2D array
        # The size of this matrix will be (image height x image width, 3) where the 3 is for the u, v, and depth
        pixels = createPixelCoordinateMatrix(depthData)

        # Define angle of rotation around x and y (not z)
        # For some reason, the x rotation is actually the y-rotation based off experiments. Guru believes it has to do with how the u and v coordinates are defined
        rotation_x = active_matrix_from_angle(0, np.deg2rad(rotation[1]))
        rotation_y = active_matrix_from_angle(1, np.deg2rad(rotation[0]))
        
        # Take the rotation matrix and use Rodrigues's formula. Needed for cv2.projectPoints
        rotationRodrigues, _ = cv2.Rodrigues(rotation_x.dot(rotation_y))
        
        # The translation is set to 0 always. Autotranslation is done after cv2.projectPoints
        translation = np.array([0, 0, 0], dtype=np.float64)
        
        # Calculate the world coordinates of the pixels
        worldGrid = calculateWorldCoordinates(pixels, cameraIntrinsicMatrix)

        # Apply cv2.projectPoints to the world coordinates to get the new pixel coordinates
        projectedImage, _ = cv2.projectPoints(worldGrid, rotationRodrigues, translation, cameraIntrinsicMatrix, distortionCoeffients)
        projectedImage = projectedImage[:, 0, :]
        # If autoTranslate is true, then we should apply it to the image
        if self.autoTranslate:
            Tx, Ty = solveForTxTy(self.pointForAutoTranslate, rotation[1], rotation[0], depthData, cameraIntrinsicMatrix)
            projectedImage[:, 0] += Tx
            projectedImage[:, 1] += Ty
            
        # Create the new RGB image
        originalPixels = pixels[:, :-1]
        newImage = createNewImage(projectedImage, originalPixels, image)
        
        return newImage