import cv2
import math
import glob
import os
import sys
import numpy as np
import tensorflow as tf

from pytransform3d.rotations import active_matrix_from_angle
from src.openpose_feature_extraction.generate_mediapipe_features import extract_mediapipe_features
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from pyk4a import PyK4APlayback
from tqdm import tqdm

# Adds the src folder to the path so generate_mediapipe_features.py can be imported
sys.path.append(os.path.abspath('../'))

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
        useBodyPixModel {int} -- which bodypix model to use to clean the depth map
        medianBlurKernelSize {int} -- kernel size for median blur (default: {5})
        guassianBlurKernelSize {int} -- kernel size for gaussian blur (default: {55})

    Returns:
        np.ndarray -- cleaned depth map
    """
    # Interesting thing to note: From visual inspections, the mean of the original depth map is really close to the depth of the body
    # A different filtering mechanism if body segmentation is used
    if type(useBodyPixModel) == 'int' and useBodyPixModel in bodyPixModelsDict:
        bodypixModel = load_model(download_model(
            bodyPixModelsDict[useBodyPixModel]))
        result = bodypixModel.predict_single(image)
        mask = result.get_mask(threshold=0.5).numpy().astype(np.uint8)[:, :, 0]
        body = depthMap[mask == 1]
        body = cv2.dilate(body, np.ones((5, 5), np.uint8), iterations=1)[:, 0]
        body[body == 0] = np.mean(body)
        depthMap[mask == 1] = body

        no_body = depthMap[mask == 0]
        no_body[no_body == 0] = 2130
        depthMap[mask == 0] = no_body

    # Temporary solution to depth filtering without body segmentation
    else:
        depthMap[depthMap == 0] = np.max(depthMap)

    # Overall filters applied to the depth map to smooth out replacements done in the previous step
    depth = cv2.GaussianBlur(cv2.medianBlur(
        depthMap, medianBlurKernelSize), (gaussianBlurKernelSize, gaussianBlurKernelSize), 0)

    return depth


def calculateWorldCoordinates(threeDPoints, cameraIntrinsicMatrix) -> np.ndarray:
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
    For more info, check the Sign-Recognition channel or ask Guru

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

    T_x = (((u_int - c_x)/f_x) * Z_c) - \
        (X_int * math.cos(theta_y)) - (Z_int * math.sin(theta_y))
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
    mask_image_grid_cv = (projectedImage[:, 0] > 0) & (projectedImage[:, 1] > 0) & (
        projectedImage[:, 0] < image.shape[0] - 1) & (projectedImage[:, 1] < image.shape[1] - 1)
    image_grid_cv = projectedImage[mask_image_grid_cv]
    original_pixels = originalPixels[mask_image_grid_cv]

    # Convert both arrays to integer values because integer values are needed for NumPy slicing
    image_grid_cv = np.around(image_grid_cv)
    image_grid_cv = image_grid_cv.astype('int')
    original_pixels = np.around(original_pixels)
    original_pixels = original_pixels.astype('int')

    # Copy the values from old to new
    newImage[image_grid_cv[:, 0], image_grid_cv[:, 1],
             :] = image[original_pixels[:, 0], original_pixels[:, 1], :]

    # Apply morphology to the new image to get rid of black spots surrounded by RGB values
    newImage = cv2.morphologyEx(
        newImage, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    return newImage


def createPixelCoordinateMatrix(depthData) -> np.ndarray:
    """createPixelCoordinateMatrix takes the pixel data and creates voxels with the depth data

    Arguments:
        depthData {np.ndarray} -- the depth data of the current frame

    Returns:
        np.ndarray -- voxel coordinates
    """
    pixels = np.indices(depthData.shape)
    rows = pixels[0].flatten().reshape(-1, 1)
    cols = pixels[1].flatten().reshape(-1, 1)
    pixels = np.hstack((rows, cols))
    flattenDepth = depthData.flatten().reshape(-1, 1)
    pixels = np.hstack((pixels, flattenDepth))

    return pixels


def get3DMediapipeCoordinates(video) -> list:
    """get3DMediapipeCoordinates get the mediapipe coordinates (non-normalized) and their actual depth according to the Azure Kinect Depth Camera

    Arguments:
        videos {list} -- list of strings containing the video paths

    Returns:
        list -- all the pose points (represented by a Nx3 NumPy Array) for each 
    """
    # Iterate over the considered videos
    for video in videos:
        # Open the depth and the color videos
        playbackImage = cv2.VideoCapture(video)
        playbackDepth = PyK4APlayback(video)
        playbackDepth.open()

        cameraIntrinsicMatrix = playbackDepth.calibration.get_camera_matrix(
            camera=1)

        currVideo = []

        while playbackImage.isOpened():
            # Get color image and the depth image for each frame
            ret, frame = playbackImage.read()
            if not ret:
                break
            capture = playbackDepth.get_next_capture()
            depth = capture.transformed_depth

            currFrame = []

            # Get the openpose data in NumPy arrays
            currMediapipeFeatures = extract_mediapipe_features(
                [frame], normalize_xy=False, save_filepath=None)
            hand0Features = np.array(
                list(currMediapipeFeatures[0]['landmarks'][0].values()))
            hand1Features = np.array(
                list(currMediapipeFeatures[0]['landmarks'][1].values()))
            poseFeatures = np.array(
                list(currMediapipeFeatures[0]['pose'].values()))

            # Only add to currFrame if the hand and body are detected
            if len(hand0Features) > 0:
                currFrame.append(hand0Features)
            if len(hand1Features) > 0:
                currFrame.append(hand1Features)
            if len(poseFeatures) > 0:
                currFrame.append(poseFeatures)

            currFrame = np.vstack(currFrame)

            # Get the depth of each point, convert depth values to meters, and add the depth values back to currVideoFrames
            depthValues = np.apply_along_axis(lambda point: getNonZeroDepth(
                int(point[0]), int(point[1]), depth), 1, currFrame).reshape(-1, 1)
            depthValues = depthValues / 1000
            currFrame = np.hstack((currFrame, depthValues))

            # Get the world coordinates
            currFrame = calculateWorldCoordinates(
                currFrame, cameraIntrinsicMatrix)

            # Add this to the list containing all the frames of the current video
            currVideo.append(currFrame)

        allVideos.append(currVideo)
        allCameraIntrinsicMatrices.append(cameraIntrinsicMatrix)

    return allVideos, allCameraIntrinsicMatrices

    # Open the depth and the color videos
    playbackImage = cv2.VideoCapture(video)
    playbackDepth = PyK4APlayback(video)
    playbackDepth.open()

    cameraIntrinsicMatrix = playbackDepth.calibration.get_camera_matrix(
        camera=1)

    currVideo = []

    while playbackImage.isOpened():
        # Get color image and the depth image for each frame
        ret, frame = playbackImage.read()
        if not ret:
            break
        capture = playbackDepth.get_next_capture()
        depth = capture.transformed_depth

        currFrame = []

        # Get the openpose data in NumPy arrays
        currMediapipeFeatures = extract_mediapipe_features(
            [frame], normalize_xy=False, save_filepath=False)
        hand0Features = np.array(
            list(currMediapipeFeatures[0]['landmarks'][0].values()))
        hand1Features = np.array(
            list(currMediapipeFeatures[0]['landmarks'][1].values()))
        poseFeatures = np.array(
            list(currMediapipeFeatures[0]['pose'].values()))

        # Only add to currFrame if the hand and body are detected
        if len(hand0Features) > 0:
            currFrame.append(hand0Features)
        if len(hand1Features) > 0:
            currFrame.append(hand1Features)
        if len(poseFeatures) > 0:
            currFrame.append(poseFeatures)

        if len(currFrame) < 1:
            continue
        
        currFrame = np.vstack(currFrame)

        # Get the depth of each point, convert depth values to meters, and add the depth values back to currVideoFrames
        depthValues = np.apply_along_axis(lambda point: getNonZeroDepth(
            int(point[1]), int(point[0]), depth), 1, currFrame).reshape(-1, 1)
        depthValues = depthValues / 1000
        currFrame = np.hstack((currFrame, depthValues))

        # Get the world coordinates
        currFrame = calculateWorldCoordinates(
            currFrame, cameraIntrinsicMatrix)

        # Add this to the list containing all the frames of the current video
        currVideo.append(currFrame)

    return currVideo, cameraIntrinsicMatrix


def getNonZeroDepth(row, col, depth) -> float:
    """getNonZeroDepth gets the non-zero depth value of a point
    This approach differs from the "cleanDepthMap" method. This method works on cleaning the depth for one point. cleanDepthMap cleans up the whole entire depth map. 
    This method works best for just a single point since doing this process for the entire depth map is computationally expensive.
    This method is also supposed to only handle cases where the pose coordinate has a 0-depth voxel, which is considered as "invalid" by the Azure Kinect SDK.

    Arguments:
        row {int} -- the row index of the point
        col {int} -- the column index of the point
        depth {np.ndarray} -- the depth map of the frame

    Returns:
        float -- the non-zero depth at a specific pixel
    """
    if depth[row, col] != 0:
        return depth[row, col]

    # Initial values
    percentNonZero = 0
    radius = 0
    # We want the area considered to have a majority of non-zero depth values
    # We keep increasing the radius until we have a majority of non-zero depth values
    while percentNonZero < 0.5 and radius < depth.shape[0] and radius < depth.shape[1]:
        minRow = row - radius if row - radius > 0 else 0
        maxRow = row + radius if row + \
            radius < depth.shape[0] else depth.shape[0]
        minCol = col - radius if col - radius > 0 else 0
        maxCol = col + radius if col + \
            radius < depth.shape[1] else depth.shape[1]

        surroundingPoints = depth[minRow:maxRow, minCol:maxCol]
        percentNonZero = np.count_nonzero(
            surroundingPoints) / ((maxRow - minRow + 1) * (maxCol - minCol + 1))

        radius += 1

    # Consider the points where the depth is not zero
    nonZeroPoints = surroundingPoints[surroundingPoints != 0]

    # Return the average of the non-zero points
    return np.mean(nonZeroPoints)


def rotation_v_0(X_int, Y_int, Z_int, cameraIntrinsicMatrix) -> float:
    """rotation_v_0 calculates the minimum rotation going left

    Arguments:
        X_int {float} -- the world X coordinate of the point
        Y_int {float} -- the world Y coordinate of the point
        Z_int {float} -- the world Z coordinate of the point
        cameraIntrinsicMatrix {np.ndarray} -- 3x3 camera matrix

    Returns:
        float -- minimum rotation going left
    """
    f_y = cameraIntrinsicMatrix[1, 1]
    c_y = cameraIntrinsicMatrix[1, 2]

    numerator = (Z_int * c_y / f_y) + Y_int
    denominator = (-1 * Y_int * c_y / f_y) + Z_int
    theta_x_radians = np.arctan(numerator / denominator)
    theta_x_degrees = np.rad2deg(theta_x_radians)

    return theta_x_degrees


def rotation_v_2160(X_int, Y_int, Z_int, cameraIntrinsicMatrix) -> float:
    """rotation_v_2160 calculates the minimum rotation going right

    Arguments:
        X_int {float} -- the world X coordinate of the point
        Y_int {float} -- the world Y coordinate of the point
        Z_int {float} -- the world Z coordinate of the point
        cameraIntrinsicMatrix {np.ndarray} -- 3x3 camera matrix

    Returns:
        float -- minimum rotation going right
    """
    f_y = cameraIntrinsicMatrix[1, 1]
    c_y = cameraIntrinsicMatrix[1, 2]

    numerator = (-1 * Z_int * (2160 - c_y) / f_y) + Y_int
    denominator = (Y_int * (2160 - c_y) / f_y) + Z_int
    theta_x_radians = np.arctan(numerator / denominator)
    theta_x_degrees = np.rad2deg(theta_x_radians)

    return theta_x_degrees


def rotation_u_0(X_int, Y_int, Z_int, cameraIntrinsicMatrix) -> float:
    """rotation_u_0 calculates the minimum rotation going up

    Arguments:
        X_int {float} -- the world X coordinate of the point
        Y_int {float} -- the world Y coordinate of the point
        Z_int {float} -- the world Z coordinate of the point
        cameraIntrinsicMatrix {np.ndarray} -- 3x3 camera matrix

    Returns:
        float -- minimum rotation going up
    """
    f_x = cameraIntrinsicMatrix[0, 0]
    c_x = cameraIntrinsicMatrix[0, 2]

    numerator = (-1 * Z_int * (0 - c_x) / f_x) + X_int
    denominator = (X_int * (0 - c_x) / f_x) + Z_int
    theta_x_radians = np.arctan(numerator / denominator)
    theta_x_degrees = np.rad2deg(theta_x_radians)

    return theta_x_degrees


def rotation_u_3840(X_int, Y_int, Z_int, cameraIntrinsicMatrix) -> float:
    """rotation_u_3840 calculates the minimum rotation going down

    Arguments:
        X_int {float} -- the world X coordinate of the point
        Y_int {float} -- the world Y coordinate of the point
        Z_int {float} -- the world Z coordinate of the point
        cameraIntrinsicMatrix {np.ndarray} -- 3x3 camera matrix

    Returns:
        float -- minimum rotation going down
    """
    f_x = cameraIntrinsicMatrix[0, 0]
    c_x = cameraIntrinsicMatrix[0, 2]

    numerator = (-1 * Z_int * (2160 - c_x) / f_x) + X_int
    denominator = (X_int * (2160 - c_x) / f_x) + Z_int
    theta_x_radians = np.arctan(numerator / denominator)
    theta_x_degrees = np.rad2deg(theta_x_radians)

    return theta_x_degrees

def augmentFrame(image, capture, rotation, cameraIntrinsicMatrix, distortionCoeffients, useBodyPixModel, medianBlurKernelSize, gaussianBlurKernelSize, autoTranslate, pointForAutoTranslate) -> np.ndarray:
    """augmentFrame rotates the current frame by the given rotation
    Arguments:
        image {np.ndarray} -- RGB image of the current frame
        capture {pyk4a.capture} -- A capture object containing the depth data for the current frame
        rotation {list} -- list of tuple containing X and Y rotation to apply to the video
        cameraIntrinsicMatrix {np.ndarray} -- 3x3 matrix explaining focal length, principal point, and aspect ratio of the camera
        distortionCoeffients {np.ndarray} -- 1x8 matrix explaining the distortion of the camera

    Returns:
        np.ndarray -- RGB projected image of the current frame given the rotation
    """
    # Clean the depth map and divide by 1000 to convert millimeters to meters
    depth = cleanDepthMap(capture.transformed_depth, image, useBodyPixModel,
                            medianBlurKernelSize, gaussianBlurKernelSize)
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
    projectedImage, _ = cv2.projectPoints(
        worldGrid, rotationRodrigues, translation, cameraIntrinsicMatrix, distortionCoeffients)
    projectedImage = projectedImage[:, 0, :]

    # If autoTranslate is true, then we should apply it to the image
    if autoTranslate:
        Tx, Ty = solveForTxTy(
            pointForAutoTranslate, rotation[1], rotation[0], depthData, cameraIntrinsicMatrix)
        projectedImage[:, 0] += Tx
        projectedImage[:, 1] += Ty

    # Create the new RGB image
    originalPixels = pixels[:, :-1]
    newImage = createNewImage(projectedImage, originalPixels, image)

    return newImage

