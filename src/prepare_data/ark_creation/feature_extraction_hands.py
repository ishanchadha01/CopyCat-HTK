import json
import numpy as np
import pandas as pd


def feature_extraction_hands(input_filepath: str, feature_names: list[str], drop_na: bool = True) -> pd.DataFrame:
  # for each of 21 landmarks of each hand, add landmark and landmark delta for each frame

  with open(input_filepath, 'r') as in_file:
    data = json.load(in_file)
 
  features_every_frame = []
  for frame_num in data:
    landmarks = data[frame_num]['landmarks']
    
    #process left, then right
    static = np.zeros((42,3))
    for feature in landmarks['0']:
      static[int(feature)] = landmarks['0'][feature]

    for feature in landmarks['1']:
      static[int(feature)+21] = landmarks['1'][feature]

    if type(features_every_frame) == list:
      features_every_frame = np.array([static.flatten()])
    else:
      features_every_frame = np.append(features_every_frame, [static.flatten()], axis=0)

  #create deltas
  deltas = features_every_frame[1:] - features_every_frame[:-1]
  deltas = np.vstack((np.zeros(deltas.shape[1]), deltas))
  features_every_frame = np.hstack((features_every_frame, deltas))
  df = pd.DataFrame(features_every_frame, columns=feature_names)
  return df

feature_extraction_hands('/media/ishan/ISHAN1/ccg/mediapipe/hands/4a.6001-singlesign/after/4a.6001-after-2022_12_14_15_11_39.630-0.data', [
      "right_landmark_0_x",
      "right_landmark_0_y",
      "right_landmark_0_z",
      "right_landmark_1_x",
      "right_landmark_1_y",
      "right_landmark_1_z",
      "right_landmark_2_x",
      "right_landmark_2_y",
      "right_landmark_2_z",
      "right_landmark_3_x",
      "right_landmark_3_y",
      "right_landmark_3_z",
      "right_landmark_4_x",
      "right_landmark_4_y",
      "right_landmark_4_z",
      "right_landmark_5_x",
      "right_landmark_5_y",
      "right_landmark_5_z",
      "right_landmark_6_x",
      "right_landmark_6_y",
      "right_landmark_6_z",
      "right_landmark_7_x",
      "right_landmark_7_y",
      "right_landmark_7_z",
      "right_landmark_8_x",
      "right_landmark_8_y",
      "right_landmark_8_z",
      "right_landmark_9_x",
      "right_landmark_9_y",
      "right_landmark_9_z",
      "right_landmark_10_x",
      "right_landmark_10_y",
      "right_landmark_10_z",
      "right_landmark_11_x",
      "right_landmark_11_y",
      "right_landmark_11_z",
      "right_landmark_12_x",
      "right_landmark_12_y",
      "right_landmark_12_z",
      "right_landmark_13_x",
      "right_landmark_13_y",
      "right_landmark_13_z",
      "right_landmark_14_x",
      "right_landmark_14_y",
      "right_landmark_14_z",
      "right_landmark_15_x",
      "right_landmark_15_y",
      "right_landmark_15_z",
      "right_landmark_16_x",
      "right_landmark_16_y",
      "right_landmark_16_z",
      "right_landmark_17_x",
      "right_landmark_17_y",
      "right_landmark_17_z",
      "right_landmark_18_x",
      "right_landmark_18_y",
      "right_landmark_18_z",
      "right_landmark_19_x",
      "right_landmark_19_y",
      "right_landmark_19_z",
      "right_landmark_20_x",
      "right_landmark_20_y",
      "right_landmark_20_z",
      "left_landmark_0_x",
      "left_landmark_0_y",
      "left_landmark_0_z",
      "left_landmark_1_x",
      "left_landmark_1_y",
      "left_landmark_1_z",
      "left_landmark_2_x",
      "left_landmark_2_y",
      "left_landmark_2_z",
      "left_landmark_3_x",
      "left_landmark_3_y",
      "left_landmark_3_z",
      "left_landmark_4_x",
      "left_landmark_4_y",
      "left_landmark_4_z",
      "left_landmark_5_x",
      "left_landmark_5_y",
      "left_landmark_5_z",
      "left_landmark_6_x",
      "left_landmark_6_y",
      "left_landmark_6_z",
      "left_landmark_7_x",
      "left_landmark_7_y",
      "left_landmark_7_z",
      "left_landmark_8_x",
      "left_landmark_8_y",
      "left_landmark_8_z",
      "left_landmark_9_x",
      "left_landmark_9_y",
      "left_landmark_9_z",
      "left_landmark_10_x",
      "left_landmark_10_y",
      "left_landmark_10_z",
      "left_landmark_11_x",
      "left_landmark_11_y",
      "left_landmark_11_z",
      "left_landmark_12_x",
      "left_landmark_12_y",
      "left_landmark_12_z",
      "left_landmark_13_x",
      "left_landmark_13_y",
      "left_landmark_13_z",
      "left_landmark_14_x",
      "left_landmark_14_y",
      "left_landmark_14_z",
      "left_landmark_15_x",
      "left_landmark_15_y",
      "left_landmark_15_z",
      "left_landmark_16_x",
      "left_landmark_16_y",
      "left_landmark_16_z",
      "left_landmark_17_x",
      "left_landmark_17_y",
      "left_landmark_17_z",
      "left_landmark_18_x",
      "left_landmark_18_y",
      "left_landmark_18_z",
      "left_landmark_19_x",
      "left_landmark_19_y",
      "left_landmark_19_z",
      "left_landmark_20_x",
      "left_landmark_20_y",
      "left_landmark_20_z",
      "delta_right_landmark_0_x",
      "delta_right_landmark_0_y",
      "delta_right_landmark_0_z",
      "delta_right_landmark_1_x",
      "delta_right_landmark_1_y",
      "delta_right_landmark_1_z",
      "delta_right_landmark_2_x",
      "delta_right_landmark_2_y",
      "delta_right_landmark_2_z",
      "delta_right_landmark_3_x",
      "delta_right_landmark_3_y",
      "delta_right_landmark_3_z",
      "delta_right_landmark_4_x",
      "delta_right_landmark_4_y",
      "delta_right_landmark_4_z",
      "delta_right_landmark_5_x",
      "delta_right_landmark_5_y",
      "delta_right_landmark_5_z",
      "delta_right_landmark_6_x",
      "delta_right_landmark_6_y",
      "delta_right_landmark_6_z",
      "delta_right_landmark_7_x",
      "delta_right_landmark_7_y",
      "delta_right_landmark_7_z",
      "delta_right_landmark_8_x",
      "delta_right_landmark_8_y",
      "delta_right_landmark_8_z",
      "delta_right_landmark_9_x",
      "delta_right_landmark_9_y",
      "delta_right_landmark_9_z",
      "delta_right_landmark_10_x",
      "delta_right_landmark_10_y",
      "delta_right_landmark_10_z",
      "delta_right_landmark_11_x",
      "delta_right_landmark_11_y",
      "delta_right_landmark_11_z",
      "delta_right_landmark_12_x",
      "delta_right_landmark_12_y",
      "delta_right_landmark_12_z",
      "delta_right_landmark_13_x",
      "delta_right_landmark_13_y",
      "delta_right_landmark_13_z",
      "delta_right_landmark_14_x",
      "delta_right_landmark_14_y",
      "delta_right_landmark_14_z",
      "delta_right_landmark_15_x",
      "delta_right_landmark_15_y",
      "delta_right_landmark_15_z",
      "delta_right_landmark_16_x",
      "delta_right_landmark_16_y",
      "delta_right_landmark_16_z",
      "delta_right_landmark_17_x",
      "delta_right_landmark_17_y",
      "delta_right_landmark_17_z",
      "delta_right_landmark_18_x",
      "delta_right_landmark_18_y",
      "delta_right_landmark_18_z",
      "delta_right_landmark_19_x",
      "delta_right_landmark_19_y",
      "delta_right_landmark_19_z",
      "delta_right_landmark_20_x",
      "delta_right_landmark_20_y",
      "delta_right_landmark_20_z",
      "delta_left_landmark_0_x",
      "delta_left_landmark_0_y",
      "delta_left_landmark_0_z",
      "delta_left_landmark_1_x",
      "delta_left_landmark_1_y",
      "delta_left_landmark_1_z",
      "delta_left_landmark_2_x",
      "delta_left_landmark_2_y",
      "delta_left_landmark_2_z",
      "delta_left_landmark_3_x",
      "delta_left_landmark_3_y",
      "delta_left_landmark_3_z",
      "delta_left_landmark_4_x",
      "delta_left_landmark_4_y",
      "delta_left_landmark_4_z",
      "delta_left_landmark_5_x",
      "delta_left_landmark_5_y",
      "delta_left_landmark_5_z",
      "delta_left_landmark_6_x",
      "delta_left_landmark_6_y",
      "delta_left_landmark_6_z",
      "delta_left_landmark_7_x",
      "delta_left_landmark_7_y",
      "delta_left_landmark_7_z",
      "delta_left_landmark_8_x",
      "delta_left_landmark_8_y",
      "delta_left_landmark_8_z",
      "delta_left_landmark_9_x",
      "delta_left_landmark_9_y",
      "delta_left_landmark_9_z",
      "delta_left_landmark_10_x",
      "delta_left_landmark_10_y",
      "delta_left_landmark_10_z",
      "delta_left_landmark_11_x",
      "delta_left_landmark_11_y",
      "delta_left_landmark_11_z",
      "delta_left_landmark_12_x",
      "delta_left_landmark_12_y",
      "delta_left_landmark_12_z",
      "delta_left_landmark_13_x",
      "delta_left_landmark_13_y",
      "delta_left_landmark_13_z",
      "delta_left_landmark_14_x",
      "delta_left_landmark_14_y",
      "delta_left_landmark_14_z",
      "delta_left_landmark_15_x",
      "delta_left_landmark_15_y",
      "delta_left_landmark_15_z",
      "delta_left_landmark_16_x",
      "delta_left_landmark_16_y",
      "delta_left_landmark_16_z",
      "delta_left_landmark_17_x",
      "delta_left_landmark_17_y",
      "delta_left_landmark_17_z",
      "delta_left_landmark_18_x",
      "delta_left_landmark_18_y",
      "delta_left_landmark_18_z",
      "delta_left_landmark_19_x",
      "delta_left_landmark_19_y",
      "delta_left_landmark_19_z",
      "delta_left_landmark_20_x",
      "delta_left_landmark_20_y",
      "delta_left_landmark_20_z"
    ])