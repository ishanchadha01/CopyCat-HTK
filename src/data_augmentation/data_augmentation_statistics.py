'''
What to do

1) Create an object in data augmentation
2) Do openpose on the original image and calculate the openpose keypoints.
3) Then project the keypoints given the rotation and get those pixel coordinates
4) Add the new image pixel coordinates compared to the ground truth pixel coordinates (ground truth is what we just did) to a data structure
5) Make a pandas dataframe from that data structure and calculate statistics
6) Statistics:
    - Avg % Error for each openpose keypoint
    - Standard Deviation for each openpose keypoint
    - Max, Min, Error for each openpose keypoint

'''

'''
What data structure should look like:

{
    "Video Name/Path": {
        "pose_1_actual": [(x1,y2) ... (xn, yn)]  # n is number of frames in video
        "pose_1_ground": [(x1,y2) ... (xn, yn)
        ...
    }
}

'''
import pandas as pd

# Change these imports to match those in the actual pipeline
from mediapipe_features import extractMediapipeFeatures
from kinect_features import extractKinectFeatures
from alpha_pose_features import extractAlphaPoseFeatures


class DataAugmentationStatistics():
    def __init__(self, csvSavePath='.', useKinectPose=True, useMediapipePose=True, useAlphaPosePose=True):
        if not useKinectPose or useMediapipePose or useAlphaPosePose:
            raise ValueError("At least one of the pose estimation methods must be used")
        
        self.kinectData = {} if useKinectPose else None
        self.mediapipeData = {} if useMediapipePose else None
        self.alphaposeData = {} if useAlphaPosePose else None
        
        self.csvSavePath = csvSavePath
        
    def addKinectPoseCoordinates(self, kinectFeatures):
        # Parse kinect pose coordinates
        pass
    
    def addMediapipePoseCoordinates(self, mediapipeFeatures):
        # parse mediapipe pose cordinates
        pass
    
    def addAlphaPosePoseCoordinates(self, alphaPoseFeatures):
        # parse alpha pose cordinates
        pass
    
    def addGroundTruthPoseCoordinates(self, image, depth, rotation) -> None:
        if self.kinectData is not None:
            self.kinectData = self.calculateGroundTruthPoseCoordinatesForKinect(self.kinectData, image, depth, rotation)
        if self.mediapipeData is not None:
            self.mediapipeData = self.calculateGroundTruthPoseCoordinatesForMediapipe(self.mediapipeData, image, depth, rotation)
        if self.alphaposeData is not None:
            self.alphaposeData = self.calculateGroundTruthPoseCoordinatesForAlphaPose(self.alphaposeData, image, depth, rotation)
        
    def calculateGroundTruthPoseCoordinatesForKinect(self, kinectData: dict, image, depth, rotation) -> dict:
        pass

    def calculateGroundTruthPoseCoordinatesForMediapipe(self, mediapipeData: dict, image, depth, rotation) -> dict:
        pass
    
    def calculateGroundTruthPoseCoordinatesForAlphaPose(self, alphaposeData: dict, image, depth, rotation) -> dict:
        pass
    
    def createStatistics(self) -> None:
        if self.kinectData is not None:
            self.kinectDataFrame = self.createStatisticsForKinect(self.kinectData)
        if self.mediapipeData is not None:
            self.mediapipeDataFrame = self.createStatisticsForMediapipe(self.mediapipeData)
        if self.alphaposeData is not None:
            self.alphaposeDataFrame = self.createStatisticsForAlphaPose(self.alphaposeData)
    
    def createStatisticsForKinect(self, kinectData: dict) -> pd.DataFrame:
        return None
    
    def createStatisticsForMediapipe(self, mediapipeData: dict) -> pd.DataFrame:
        return None
    
    def createStatisticsForAlphaPose(self, alphaposeData: dict) -> pd.DataFrametr:
        return None
    
    def saveStatistics(self):
        # Path should be csvStatiscs/kinect, etc and make it different for each pose
        pass