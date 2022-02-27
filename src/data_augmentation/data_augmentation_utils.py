import cv2
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import math
import numpy as np
import glob
import os

# A dictionary that defines which body pixel model to use
bodyPixModelsDict = {
    0: BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_8,
    1: BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16,
    2: BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_8,
    3: BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_16,
    4: BodyPixModelPaths.MOBILENET_FLOAT_100_STRIDE_8,
    5: BodyPixModelPaths.MOBILENET_FLOAT_100_STRIDE_16,
    6: BodyPixModelPaths.RESNET50_FLOAT_STRIDE_16,
    7: BodyPixModelPaths.RESNET50_FLOAT_STRIDE_32
}

def countFrames(video) -> int:
    """countFrames counts the number of frames in a video

    Arguments:
        video {str} -- path of video for which the frames are to be counted

    Returns:
        int -- number of frames in the video
    """
    try:
        # If cv2 version greater than 3, then metadata is different
        if int(cv2.__version__.split('.')[0]) > 3:
            frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            frameCount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    # Count frames manually if this fails
    except:
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

def cleanDepthMap(depthMap, image, useBodyPixModel, medianBlurKernelSize=5, gaussianBlurKernelSize=55) -> np.ndarray:
    """cleanDepthMap processes the depth map to remove the 0 depth pixels and replace them

    Arguments:
        depthMap {np.ndarray} -- the depth map to clean
        image {np.ndarray} -- image to process using bodypix model
        useBodyPixModel {bool} -- whether or not to use the bodypix model to clean the depth map (default: {True})
        medianBlurKernelSize {int} -- kernel size for median blur (default: {5})
        guassianBlurKernelSize {int} -- kernel size for gaussian blur (default: {55})
        
    Returns:
        np.ndarray -- cleaned depth map
    """
    # Interesting thing to note: From visual inspections, the mean of the original depth map is really close to the depth of the body  
    # A different filtering mechanism if body segmentation is used
    if type(useBodyPixModel) == 'int' & useBodyPixModel in bodyPixModelsDict:
        bodypixModel = load_model(download_model(bodyPixModelsDict[useBodyPixModel]))
        result = bodypixModel.predict_single(image)
        mask = result.get_mask(threshold=0.5).numpy().astype(np.uint8)[:, :, 0]
        body = depthMap[mask == 1]
        body = cv2.dilate(body, np.ones((5,5),np.uint8), iterations=1)[:, 0]
        body[body == 0] = np.mean(body)
        depthMap[mask == 1] = body

        no_body = depthMap[mask == 0]
        no_body[no_body == 0] = 2130
        depthMap[mask == 0] = no_body

    # Temporary solution to depth filtering without body segmentation
    else:
        depthMap[depthMap == 0] = np.max(depthMap)   
    
    # Overall filters applied to the depth map to smooth out replacements done in the previous step
    depth = cv2.GaussianBlur(cv2.medianBlur(depthMap, medianBlurKernelSize), (gaussianBlurKernelSize, gaussianBlurKernelSize), 0)
    
    return depth

def calculateWorldCoordinates(threeDPoints, cameraIntrinsicMatrix):
    """calculateWorldCoordinates calculates the world coordinates (3D points) from the 2D array

    Arguments:
        threeDPoints {np.ndarray} -- Nx3 array of 2D points with depth as the last column
        cameraIntrinsicMatrix {np.ndarray} -- camera intrinsic matrix

    Returns:
        np.ndarray -- world coordinates (3D points)
    """
    depthData = np.copy(threeDPoints[:, -1]).reshape(-1, 1)
    threeDPoints[:, -1] = 1
    worldGrid = np.linalg.inv(cameraIntrinsicMatrix) @ threeDPoints.transpose()
    worldGrid = worldGrid.T * depthData
    
    return worldGrid

def getListVideos(datasetFolder) -> list:
    """getListVideos gets the list of the paths of all the .mkv videos in the dataset folder

    Returns:
        list -- The list of the paths of all the .mkv videos in the dataset folder
    """
    # Scan for every .mkv file in the dataset folder (.mkv files have depth, ir, and RGB data and are used by the Azure Kinect SDK)
    result = []
    for folder in os.walk(datasetFolder):
        for video in glob.glob(os.path.join(folder[0], '*.mkv')):
            result.append(video)
    return result

def solveForTxTy(pointForAutoTranslate, y_rotation, x_rotation, depthMap, cameraIntrinsicMatrix) -> tuple:
        """solveForTxTy uses math calculations to solve for the translation of the image based on the given point for auto translation

        Arguments:
            pointForAutoTranslate {tuple} -- tuple containing the x and y coordinates of the point to auto translate
            y_rotation {float} -- the given y rotation
            x_rotation {float} -- the given x rotation
            depthMap {np.ndarray} -- the depth map of the current frame
            cameraIntrinsicMatrix {np.ndarray} -- the camera intrinsic matrix

        Returns:
            tuple -- the X, Y translation of the image needed for centering in pixels
        """
        u_int, v_int = pointForAutoTranslate
        
        f_x = cameraIntrinsicMatrix[0, 0]
        f_y = cameraIntrinsicMatrix[1, 1]
        c_x = cameraIntrinsicMatrix[0, 2]
        c_y = cameraIntrinsicMatrix[1, 2]
        Z_int = depthMap[u_int, v_int]

        X_int = (Z_int/f_x) * (u_int - c_x)
        Y_int = (Z_int/f_y) * (v_int - c_y)

        theta_x = math.radians(y_rotation)
        theta_y = math.radians(x_rotation)

        Z_c = (-1 * X_int * math.cos(theta_x) * math.sin(theta_y)) + \
                (Y_int * math.sin(theta_x)) + \
                (Z_int * math.cos(theta_x) * math.cos(theta_y))

        T_x = (((u_int - c_x)/f_x) * Z_c) - (X_int * math.cos(theta_y)) - (Z_int * math.sin(theta_y))
        T_y = (((v_int - c_y)/f_y) * Z_c) - \
                (X_int * math.sin(theta_x) * math.sin(theta_y)) - \
                (Y_int * math.cos(theta_x)) + \
                (Z_int * math.sin(theta_x) * math.cos(theta_y))

        T_x = f_x * T_x / Z_c
        T_y = f_y * T_y / Z_c

        return T_x, T_y
    
def createNewImage(projectedImage, originalPixels, image) -> np.ndarray:
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

    # Apply morphology to the new image to get rid of black spots surrounded by RGB values
    new_image = cv2.morphologyEx(new_image, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    
    return newImage

def createPixelCoordinateMatrix(depthData) -> np.ndarray:
    """createPixelCoordinateMatrix takes the pixel data and creates voxels with the depth data

    Arguments:
        depthData {np.ndarray} -- the depth data of the current frame

    Returns:
        np.ndarray -- voxel coordinates
    """
    pixels = np.indices(depthData.shape)
    rows = pixels[0].flatten().reshape(-1,1)
    cols = pixels[1].flatten().reshape(-1,1)
    pixels = np.hstack((rows, cols))
    flattenDepth = depthData.flatten().reshape(-1, 1)
    pixels = np.hstack((pixels, flattenDepth))
    
    return pixels