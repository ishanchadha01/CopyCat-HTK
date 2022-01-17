
'''
As of 1/16/2022...

If you complete a task, don't remove it, just put your name beside it!
Maybe there's a better way to do this with GitHub Issues, etc.

**************** THINGS TO DO ****************

1) Add nested progress bars
    - Guru (as of 1/16/2022) has a progress bar to monitor the progress of the data augmentation for all videos.
    - However, it is also nice to have a progress bar to monitor the progress of augmenting 1 video at a time
    - This requires have 1 main progress bar and 1 nested progress bar
    - All functions/code relating to nesting progress bars will have the comment:
      # For future use: Nested progress bars

2) Change the default dataset folder and default output folder in __init__ method

3) Extracting Kinect, Mediapipe, AlphaPose OpenPose features directly from the augmented videos
    - Right now, the script saves new .avi files
    - However, for space, this may or may not be the best way to do this
    - Instead of saving new .avi files, we could extract the OpenPose features directly from the augmented videos while augmenting
    - This way, we'd save new .json files and not new .avi files
    
4) === GURU WORKING ON THIS === Implement a Data Augmentation Statistics class to track reprojection errors in Mediapipe, Kinect, and AlphaPose
    - Might be a good idea to get Guru on this if you need help
    
5) Speed up data augmentation computations
    - Look into things like Numba, Cython and see if these can be used to speed up NumPy array computations
    - Right now, the process that takes the most time (~5 seconds per frame) is the cv2.projectPoints function
    - Is it worth implementing that ourselves and then using Numba/Cython to speed it up?
    - Also, from some reading, Guru found out that Pypy can slow down NumPy computations

6) Implement the extractMediapipeFeatures, extractKinectFeatures, and extractAlphaPoseFeatures functions (relates to #3)
    - These rely on the preexisting functions from before, but they need to return a different data structure
    - For data augmentation, I need a numpy array of the keypoints (size keypoints x 2). A list/tuple of NumPy arrays would work too depending on the OpenPose implementation used
    - You could use the functions from #3 but then add a parameter that changes the output based on what I said
    

---------------- NICE TO HAVES ----------------

1) Add translations along with rotations into the pipeline
    - Might be a good idea to get Guru on this if you need help

2) Create an Exception for if the image is clipped enough
    - If the rotations and translations cause the OpenPose points to not be present on the image, we should raise an exception for that
    - This way, bad rotations and translations don't make it into the training data
    - Thad said that as of 1/16/2022, it isn't a priority

3) Calculate the maximum rotation/translations possible that doesn't allow videos to be clipped (relates to 2)
    - Guru was working on some math for this until Thad said that as of 1/16/2022, it isn't a priority
    
4) Look into PyK4A capture's color parameter
    - When augmenting a video, OpenCV is used to look at the video and extract the color image and PyK4A is used to look at the depth image
    - Is it worth it to just use PyK4A to get both the depth map and color image? What would be the performance impacts?
    - Ask Guru for specifics

'''

'''
Basic Explanation of Data Augmentation Class
--------------------------------------------

Class for Data Augmentation
- Accepts a list of rotations

Returns:
- A list of all the paths of all the data augmented videos with the list of rotations

Steps it does:
1) Get the list of all the videos
2) For each video, apply all the rotations
3) Save a new video for each rotation
4) Add that to a list of the paths of data augmented videos
'''

# Imports
import cv2
from pyk4a import PyK4APlayback
import numpy as np
import glob
import os
from pytransform3d.rotations import active_matrix_from_angle
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from tqdm import tqdm
from data_augmentation_statistics import DataAugmentationStatistics

# Change these imports to match those in the actual pipeline (look at Things to Do #6)
from mediapipe_features import extractMediapipeFeatures
from kinect_features import extractKinectFeatures
from alpha_pose_features import extractAlphaPoseFeatures

# Load BodyPix model used for cleaning depth maps
bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

# For future use: Nested progress bars
def countFrames(video) -> int:
    """countFrames counts the number of frames in a video

    Arguments:
        video {str} -- path of video for which the frames are to be counted

    Returns:
        int -- number of frames in the video
    """
    playbackImage = cv2.VideoCapture(video)
    frameCount = 0
    while playbackImage.isOpened():
        ret, _ = playbackImage.read()
        if ret == True:
            frameCount += 1
        else:
            break
    playbackImage.release()
    return frameCount


class DataAugmentation():
    def __init__(self, rotations, datasetFolder='./CopyCatDatasetWIP', outputPath='.', medianBlurKernelSize=5, gaussianBlurKernelSize=55, useBodyPixModel=True, trackStatistics=True, createMediapipeJSON=True, createKinectJSON=True, createAlphaPoseJSON=True, csvStatisticsSavePath='.') -> None:
        """__init__ creates the Data Augmentation Object

        Arguments:
            rotations {list of tuples} -- a list of x, y rotations to apply to the videos

        Keyword Arguments:
            datasetFolder {str} -- the path where the original videos are located (default: {'./CopyCatDatasetWIP'})
            outputPath {str} -- the path where the augmented videos will be saved (default: {'.'})
            medianBlurKernelSize {int} -- kernel size for the median blur filter applied to depth maps and projected images -- from experiments, only number that works is 5 (default: {5})
            gaussianBlurKernelSize {int} -- kernel size for the gaussian blur filter applied to depth maps (default: {55})
            useBodyPixModel {bool} -- whether TensorFlow's bodypix model should be used to create better depth maps (default: {True})
            trackStatistics {bool} -- whether to track reprojection statistics of the data augmentation (default: {True})
            createMediapipeJSON {bool} -- whether to create a JSON file for Mediapipe (default: {True})
            createKinectJSON {bool} -- whether to create a JSON file for Kinect (default: {True})
            createAlphaPoseJSON {bool} -- whether to create a JSON file for AlphaPose (default: {True})
            csvStatisticsSavePath {str} -- save path for OpenPose statistics CSV (default: {'.'})

        Raises:
            NameError: If the dataset folder does not exist
            NameError: If the output path does not exist
            ValueError: If the X rotation is greater than 90 or less than 0
            ValueError: If the Y rotation is greater than 90 or less than 0
        """
        #DataAugmentationStatistics
        # If the dataset folder or the output folder does not exist, raise an error
        if not os.path.exists(datasetFolder):
            raise NameError(f'Dataset folder {datasetFolder} does not exist')
        
        if not os.path.exists(outputPath):
            raise NameError(f'Output path {outputPath} does not exist')
        
        if trackStatistics and not createMediapipeJSON and not createKinectJSON and not createAlphaPoseJSON:
            raise ValueError('Cannot track statistics without creating JSON files')
        
        if trackStatistics and not os.path.exists(csvStatisticsSavePath):
            raise NameError(f'OpenPose statistics save path {csvStatisticsSavePath} does not exist')
        
        # If the x or y angle is negative or greater than 90, raise an error
        for x, y in rotations:
            if x < 0 or x >= 90:
                raise ValueError('X Rotation must be between 0 and less than 90')
            elif y < 0 or y >= 90:
                raise ValueError('Y Rotation must be between 0 and less than 90')
        
        self.rotations = rotations
        self.datasetFolder = datasetFolder
        self.medianBlurKernelSize = medianBlurKernelSize
        self.gaussianBlurKernelSize = gaussianBlurKernelSize
        self.useBodyPixModel = useBodyPixModel
        self.outputPath = outputPath
        self.createMediapipeJSON = createMediapipeJSON
        self.createKinectJSON = createKinectJSON
        self.createAlphaPoseJSON = createAlphaPoseJSON
        
        if trackStatistics:
            self.statistics = DataAugmentationStatistics(createMediapipeJSON, createKinectJSON, createAlphaPoseJSON, csvStatisticsSavePath)
        else:
            self.statistics = None
    
    def __str__(self) -> str:
        """__str__ returns a string representation of the DataAugmentation Object when used in print statements

        Returns:
            str -- string representation of the DataAugmentation Object (when used in print statements)
        """
        retStr = "Data Augmentation With Rotations:\n"
        for rotation in self.rotations:
            retStr += f"{str(rotation)}\n"
        return retStr
    
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
        listOfVideos = self.getListVideos()
        self.pbar = tqdm(total=len(listOfVideos))
        
        # Start applying data augmentation to each video while appending the augmented video path to a new list
        newVideos = []
        for video in listOfVideos:
            for rotation in self.rotations:
                newVideos.append(self.augmentVideo(video, rotation))
                self.pbar.update(1)
        
        return newVideos
        
    def getListVideos(self) -> list:
        """getListVideos gets the list of the paths of all the .mkv videos in the dataset folder

        Returns:
            list -- The list of the paths of all the .mkv videos in the dataset folder
        """
        # Scan for every .mkv file in the dataset folder (.mkv files have depth, ir, and RGB data and are used by the Azure Kinect SDK)
        result = []
        for folder in os.walk(self.datasetFolder):
            for video in glob.glob(os.path.join(folder[0], '*.mkv')):
                result.append(video)
        return result

    def augmentVideo(self, video, rotation) -> str:
        """augmentVideo augments a video with a given rotation

        Arguments:
            video {str} -- the path of the video to augment
            rotation {list} -- list of tuple containing X and Y rotation to apply to the video

        Returns:
            str -- destination path of the augmented video
        """
        # Get the video name by getting the last str when splitting on '/'
        videoName = video.split('/')[-1]
        
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
        currVideoPath = f'./{self.outputPath}/rotated_x_{rotation[0]}_y_{rotation[1]}_{videoName}.avi'
        frameWidth = int(playbackImage.get(3))
        frameHeight = int(playbackImage.get(4))
        
        # If the augmented video exists, then there's no need to run data augmentation again. Only do this if the augmented video does not exist
        if not os.path.exists(currVideoPath):
            out = cv2.VideoWriter(currVideoPath, codec, fps, (frameWidth, frameHeight))

            # ADD NESTED PBAR HERE using the total number of frames in the video. Code to get number of frames is below
            #frameCount = self.countFrames(video)
            
            # Iterate over the video and get the projected image. Then write that image to the new video
            while (True):
                try:
                    
                    capture = playbackDepth.get_next_capture()
                    _, image = playbackImage.read()
                    
                    newImage = self.augmentFrame(image, capture, rotation, intrinsicCameraMatrix, distortionCoefficients)
                    
                    # EXTRACT OPENPOSE POINTS HERE AND SAVE TO JSON BASED ON PARAMS FROM INIT METHOD (look at Things to do 3)
                    if self.createKinectJSON:
                        kinectFeatures = extractKinectFeatures(newImage)
                        self.statistics.addKinectPoseCoordinates(kinectFeatures)
                    if self.createMediapipeJSON:
                        mediapipeFeatures = extractMediapipeFeatures(newImage)
                        self.statistics.addMediapipePoseCoordinates(mediapipeFeatures)
                    if self.createAlphaPoseJSON:
                        alphaPoseFeatures = extractAlphaPoseFeatures(newImage)
                        self.statistics.addAlphaPosePoseCoordinates(alphaPoseFeatures)
                    
                    self.statistics.addGroundTruthPoseCoordinates(image, capture.transformed_depth, rotation, intrinsicCameraMatrix, distortionCoefficients)
                    
                    out.write(cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))
                    
                except EOFError:
                    break
            
            # Close all the videos
            playbackDepth.close()
            out.release()
            playbackImage.release()
            
        return currVideoPath
    
    def augmentFrame(self, image, capture, rotation, cameraIntrinsicMatrix, distortionCoeffients) -> np.ndarray:
        """augmentFrame takes a current frame and applies the given rotation to it

        Arguments:
            image {np.ndarray} -- RGB image of the current frame
            capture {pyk4a.capture} -- A capture object containing the depth data for the current frame
            playbackDepth {pyk4a.playback} -- A playback object containing the camera calibration parameters for the current frame
            rotation {list} -- list of tuple containing X and Y rotation to apply to the video

        Returns:
            np.ndarray -- RGB projected image of the current frame given the rotation
        """
        # Clean the depth map and divide by 1000 to convert millimeters to meters
        depth = self.cleanDepthMap(capture.transformed_depth, image)
        depthData = depth / 1000
        
        # Define a matrix that contains all the pixel coordinates and their depths in a 2D array
        # The size of this matrix will be (image height x image width, 3) where the 3 is for the RGB values
        pixels = np.indices(depthData.shape)
        rows = pixels[0].flatten().reshape(-1,1)
        cols = pixels[1].flatten().reshape(-1,1)
        pixels = np.hstack((rows, cols))
        flattenDepth = depthData.flatten().reshape(-1, 1)
        pixels = np.hstack((pixels, flattenDepth))

        # Define angle of rotation around x and y (not z)
        # For some reason, the x rotation is actually the y-rotation based off experiments. Guru believes it has to do with how the u and v coordinates are defined
        rotation_x = active_matrix_from_angle(0, np.deg2rad(rotation[1]))
        rotation_y = active_matrix_from_angle(1, np.deg2rad(rotation[0]))
        
        # Take the rotation matrix and use Rodrigues's formula. Needed for cv2.projectPoints
        rotationRodrigues, _ = cv2.Rodrigues(rotation_x.dot(rotation_y))
        
        # The translation is currently a 0 vector. Look at "Nice to Haves" number 5 at the top of this class
        translation = np.array([0, 0, 0], dtype=np.float64)
        
        # Calculate the world coordinates of the pixels
        threeDPoints = pixels
        depthData = np.copy(threeDPoints[:, -1]).reshape(-1, 1)
        threeDPoints[:, -1] = 1
        worldGrid = np.linalg.inv(cameraIntrinsicMatrix) @ threeDPoints.transpose()
        worldGrid = worldGrid.T * depthData

        # Apply cv2.projectPoints to the world coordinates to get the new pixel coordinates
        projectedImage, _ = cv2.projectPoints(worldGrid, rotationRodrigues, translation, cameraIntrinsicMatrix, distortionCoeffients)

        # Create the new RGB image
        projectedImage = projectedImage[:, 0, :]
        originalPixels = pixels[:, :-1]
        newImage = self.createNewImage(projectedImage, originalPixels, image)
        
        return newImage
    
    def cleanDepthMap(self, depthMap, image) -> np.ndarray:
        """cleanDepthMap processes the depth map to remove the 0 depth pixels and replace them

        Arguments:
            depthMap {np.ndarray} -- the depth map to clean
            image {np.ndarray} -- image to process using bodypix model

        Returns:
            np.ndarray -- [description]
        """
        # Interesting thing to note: From visual inspections, the mean of the original depth map is really close to the depth of the body
        originalDepthMap = np.copy(depthMap)
        
        # A different filtering mechanism if body segmentation is used
        if self.useBodyPixModel:
            result = bodypix_model.predict_single(image)
            mask = result.get_mask(threshold=0.5).numpy().astype(np.uint8)[:, :, 0]
            body = depthMap[mask == 1]
            body[body == 0] = np.mean(body)
            depthMap[mask == 1] = body

            no_body = depthMap[mask == 0]
            no_body[no_body == 0] = np.max(originalDepthMap)
            depthMap[mask == 0] = no_body

        # Temporary solution to depth filtering without body segmentation
        else:
            depthMap[depthMap == 0] = np.max(depthMap)   
        
        # Overall filters applied to the depth map to smooth out replacements done in the previous step
        depth = cv2.GaussianBlur(cv2.medianBlur(depthMap, self.medianBlurKernelSize), (self.gaussianBlurKernelSize, self.gaussianBlurKernelSize), 0)
        
        return depth
    
    def createNewImage(self, projectedImage, originalPixels, image) -> np.ndarray:
        """createNewImage created the RGB projected image given the pixel coordinates

        Arguments:
            projectedImage {np.ndarray} -- (2160 x 3840) X 2 array containing the projected pixel coordinates in row, column format
            originalPixels {np.ndarray} -- (2160 x 3840) X 2 array containing the original pixel coordinates in row, column format
            image {np.ndarray} -- original RGB image

        Returns:
            np.ndarray -- RGB projected image
        """
        # Define the new RGB image as 0s. Helpful later on because all 0s correspond to black pixels
        newImage = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # Identify the pixels that clip and not consider them when copying the RGB values from the old to the new iamge
        mask_image_grid_cv = (projectedImage[:, 0] > 0) & (projectedImage[:, 1] > 0) & (projectedImage[:, 0] < image.shape[0] - 1) & (projectedImage[:, 1] < image.shape[1] - 1)
        image_grid_cv = projectedImage[mask_image_grid_cv]
        original_pixels = originalPixels[mask_image_grid_cv]

        # Convert both arrays to integer values because integer values are needed for NumPy slicing
        image_grid_cv = np.around(image_grid_cv)
        image_grid_cv = image_grid_cv.astype('int')
        original_pixels = np.around(original_pixels)
        original_pixels = original_pixels.astype('int')

        # Copy the values from old to new
        newImage[image_grid_cv[:, 0], image_grid_cv[:, 1], :] = image[original_pixels[:, 0], original_pixels[:, 1], :]

        # Apply a median blur to the new image to get rid of black spots surrounded by RGB values
        newImage = cv2.medianBlur(newImage, self.medianBlurKernelSize)
        
        return newImage
    
    
    
    
    