from create_elan import *
from plot_mediapipe import draw_mediapipe_landmarks
import tkinter as tk

import os
import platform 
import subprocess
import threading


class ElanGui:

  def __init__(self, video_filepath=""):
    self.window = tk.Tk()
    self.window.geometry("800x400")
    self.elan_exec = ""
    self.video_fp_list = []
    self.video_fp_list_idx = 0
    self.boundary_annotations = {}
    self.eaf_savedir = ""
    self.bad_landmarks = {}
    self.feature_name_display = []

    elan_exec_text = tk.StringVar()
    elan_exec_text.set("Path to ELAN executable")
    self.elan_exec_label = tk.Label(self.window, textvariable=elan_exec_text, height=2)
    self.elan_exec_label.grid(row=1, column=1)
    self.elan_exec_textbox = tk.Text(self.window, height=2, width=40)
    self.elan_exec_textbox.insert("end", "/home/ishan/Documents/research/ccg/elan/bin/ELAN")
    self.elan_exec_textbox.grid(row=1, column=2)

    mlf_fp_text = tk.StringVar()
    mlf_fp_text.set("MLF filepath name for annotations")
    self.mlf_fp_label = tk.Label(self.window, textvariable=mlf_fp_text, height=2)
    self.mlf_fp_label.grid(row=2, column=1)
    self.mlf_textbox = tk.Text(self.window, height=2, width=40)
    self.mlf_textbox.insert("end", "/home/ishan/Documents/research/ccg/copycat/CopyCat-HTK/projects/MediaPipe/results/res_hmm5.mlf")
    self.mlf_textbox.grid(row=2, column=2)

    macros_fp_text = tk.StringVar()
    macros_fp_text.set("Filepath where GMM means/vars per feature, state, and word are located")
    self.macros_fp_label = tk.Label(self.window, textvariable=macros_fp_text, height=2)
    self.macros_fp_label.grid(row=3, column=1)
    self.macros_textbox = tk.Text(self.window, height=2, width=40)
    self.macros_textbox.insert("end", "/home/ishan/Documents/research/ccg/copycat/CopyCat-HTK/projects/MediaPipe/models/hmm5/newMacros")
    self.macros_textbox.grid(row=3, column=2)

    feature_labels_text = tk.StringVar()
    feature_labels_text.set("JSON file with all feature names")
    self.feature_labels_label = tk.Label(self.window, textvariable=feature_labels_text, height=2)
    self.feature_labels_label.grid(row=4, column=1)
    self.feature_labels_textbox = tk.Text(self.window, height=2, width=40)
    self.feature_labels_textbox.insert("end", "/home/ishan/Documents/research/ccg/copycat/CopyCat-HTK/projects/MediaPipe/configs/features.json")
    self.feature_labels_textbox.grid(row=4, column=2)

    feature_data_text = tk.StringVar()
    feature_data_text.set("Directory with ARK files containing all extracted feature data")
    self.feature_data_label = tk.Label(self.window, textvariable=feature_data_text, height=2)
    self.feature_data_label.grid(row=5, column=1)
    self.feature_data_textbox = tk.Text(self.window, height=2, width=40)
    self.feature_data_textbox.insert("end", "/home/ishan/Documents/research/ccg/copycat/CopyCat-HTK/projects/MediaPipe/data/ark")
    self.feature_data_textbox.grid(row=5, column=2)

    video_dir_text = tk.StringVar()
    video_dir_text.set("Video directory name")
    self.video_dir_label = tk.Label(self.window, textvariable=video_dir_text, height=2)
    self.video_dir_label.grid(row=6, column=1)
    self.video_dir_textbox = tk.Text(self.window, height=2, width=40)
    self.video_dir_textbox.insert("end", "/home/ishan/Documents/research/ccg/copycat/DATA/input")
    self.video_dir_textbox.grid(row=6, column=2)

    eaf_savedir_text = tk.StringVar()
    eaf_savedir_text.set("Elan annotation file save directory")
    self.eaf_savedir_label = tk.Label(self.window, textvariable=eaf_savedir_text, height=2)
    self.eaf_savedir_label.grid(row=7, column=1)
    self.eaf_savedir_textbox = tk.Text(self.window, height=2, width=40)
    self.eaf_savedir_textbox.insert("end", "/home/ishan/Documents/research/ccg/elan")
    self.eaf_savedir_textbox.grid(row=7, column=2)

    self.toggle_landmarks = tk.IntVar()
    self.landmarks_checkbox = tk.Checkbutton(self.window, text="Show Mediapipe Landmarks", variable=self.toggle_landmarks, onvalue=1)
    self.landmarks_checkbox.grid(row=8, column=1)
    self.landmarks_textbox = tk.Text(self.window, height=2, width=40)
    self.landmarks_textbox.insert("end", "/home/ishan/Documents/research/ccg/elan")
    self.landmarks_textbox.grid(row=8, column=2)

    self.start_loop_button = tk.Button(self.window, text="Start Video Loop", command=self.launch_elan)
    self.start_loop_button.grid(row=9, column=2)

    self.next_video_button = tk.Button(self.window, text="Next Video", command=self.next_video)

    self.close_button = tk.Button(self.window, text="Quit", command=self.window.destroy)
    self.close_button.grid(row=10, column=2)


    self.window.mainloop()


  def launch_elan(self):

    # Initialize next video button, read textboxes after that
    self.start_loop_button.grid_forget()

    # Get path to ELAN executable
    self.elan_exec = self.elan_exec_textbox.get("1.0", "end-1c")
    self.elan_exec_label.grid_forget()
    self.elan_exec_textbox.grid_forget()

    # Get MLF filepath
    mlf_fp = self.mlf_textbox.get("1.0", "end-1c")
    self.boundary_annotations = mlf_to_dict(mlf_fp)
    self.mlf_fp_label.grid_forget()
    self.mlf_textbox.grid_forget()

    # Get feature labels from JSON file
    feature_labels_fp = self.feature_labels_textbox.get("1.0", "end-1c")
    self.feature_labels = json.load(open(feature_labels_fp))['selected_features']
    self.feature_labels_label.grid_forget()
    self.feature_labels_textbox.grid_forget()

    # Get directory containing feature data for each video
    self.feature_data_dir = self.feature_data_textbox.get("1.0", "end-1c")
    self.feature_data_label.grid_forget()
    self.feature_data_textbox.grid_forget()

    # Get model data from GMMs file
    macros_fp = self.macros_textbox.get("1.0", "end-1c")
    self.model_data = make_model_dict(macros_fp, self.feature_labels)
    self.macros_fp_label.grid_forget()
    self.macros_textbox.grid_forget()

    # Check whether to display landmarks for each video
    self.show_landmarks = self.toggle_landmarks.get()
    self.landmarks_dir = self.landmarks_textbox.get("1.0", "end-1c")
    self.landmarks_checkbox.grid_forget()
    self.landmarks_textbox.grid_forget()

    # Get all videos to iterate over
    video_dir = self.video_dir_textbox.get("1.0", "end-1c")
    for root, dirs, files in os.walk(video_dir):
      for video_fp in files:
        if video_fp.endswith(".mp4"):
          self.video_fp_list.append(os.path.join(root, video_fp))
    self.video_fp_list = list(set(self.video_fp_list))
    first_video = self.video_fp_list[0]
    video_len = int(float(FFProbe(first_video).video[0].duration) * 1000)
    self.boundary_annotations = scale_annotations(self.boundary_annotations, video_len)
    self.video_dir_label.grid_forget()
    self.video_dir_textbox.grid_forget()

    # Get EAF savedir
    self.eaf_savedir = self.eaf_savedir_textbox.get("1.0", "end-1c")
    self.eaf_savedir_label.grid_forget()
    self.eaf_savedir_textbox.grid_forget()

    # Create next video button and run first video
    self.next_video_button.grid(row=1, column=2)
    self.next_video()


  def next_video(self):
    # Process video at current index
    video_fp = self.video_fp_list[self.video_fp_list_idx]
    phrase = os.path.basename(os.path.splitext(video_fp)[0])
    if self.show_landmarks:
      video_fp = draw_mediapipe_landmarks(video_filepath=video_fp, new_video_filepath=os.path.join(self.landmarks_dir, phrase) + ".mp4")

    # Make elan file with annotations
    eaf_obj = make_elan(self.boundary_annotations, has_states=True, video_dirs=[video_fp], \
      eaf_savedir=self.eaf_savedir)[0]
    eaf_path = os.path.join(self.eaf_savedir, phrase) + ".eaf"

    # Get feature data from ARK
    ark_fp = os.path.join(self.feature_data_dir, phrase) + ".ark"
    feature_data = read_ark_file(ark_fp)

    # Label bad frames
    video_len = int(float(FFProbe(video_fp).video[0].duration) * 1000)
    dists = compute_frame_dists(self.model_data, feature_data, self.boundary_annotations, self.feature_labels, phrase, video_len)
    self.bad_frames, self.outliers = get_outliers(dists)
    label_worst_frames(self.bad_frames, self.outliers, eaf_path, video_len)
    
    # Create feature display
    state_name = tk.StringVar()
    state_name.set("Enter state and word names")
    self.state_name_label = tk.Label(self.window, textvariable=state_name, height=2)
    self.state_name_label.grid(row=1, column=1)
    self.word_name_textbox = tk.Text(self.window, height=2, width=20)
    self.word_name_textbox.grid(row=1, column=2)
    self.state_name_textbox = tk.Text(self.window, height=2, width=20)
    self.state_name_textbox.grid(row=1, column=3)
    self.check_state_features_button = tk.Button(self.window, text="Submit", command=lambda: self.check_state_features(video_len))
    self.check_state_features_button.grid(row=2, column=2)
    self.window.update()

    # Execute ELAN
    t = threading.Thread(target=lambda: self.execute_elan(eaf_path))
    t.daemon = True
    t.start()

    # self.process_video(eaf_obj)
    self.video_fp_list_idx += 1


  def process_video(self, eaf_obj):
    tier_names = list(eaf_obj.get_tier_names())
    video_fp = self.video_fp_list[self.video_fp_list_idx]

    # Process frames with bad landmarks
    if "bad_landmarks" in tier_names:
      tier_names.remove("bad_landmarks")
      self.bad_landmarks[video_fp] = eaf_obj.get_annotation_data_for_tier("bad_landmarks")

    # Once the video has been closed out of, reexamine EAF for new word/state boundaries
    new_annotations = {}
    for tier in tier_names:
      new_annotations[tier] = {}
      for start, end, state in eaf_obj.get_annotation_data_for_tier(tier):
        new_annotations[tier][state] = [start, end]
    self.boundary_annotations[video_fp] = new_annotations


  def check_state_features(self, video_len):
    # Check if any features are outliers, and display such features
    feature_outliers_dict = feature_outliers_by_state(self.boundary_annotations, self.outliers, self.feature_labels, video_len)
    word= self.word_name_textbox.get("1.0", "end-1c")
    state = self.state_name_textbox.get("1.0", "end-1c")

    # Clear display if new one requested
    for display in self.feature_name_display:
      display.grid_forget()

    self.feature_name_display = []
    self.close_button.grid_forget()
    for tk_row, (name, value) in enumerate(feature_outliers_dict[word][state].items()):
      feature_name = tk.StringVar()
      feature_name.set(name)
      feature_value = tk.StringVar()
      feature_value.set(value)
      feature_name_label = tk.Label(self.window, textvariable=feature_name, height=2)
      feature_name_label.grid(row=tk_row+3, column=1)
      feature_value_label = tk.Label(self.window, textvariable=feature_value, height=2)
      feature_value_label.grid(row=tk_row+3, column=2)
      self.feature_name_display.append(feature_name_label)
      self.feature_name_display.append(feature_value_label)
    self.close_button.grid(row=30, column=2)


  def execute_elan(self, eaf_path):
    # TODO: test on different OSs, possibly remove conditional
    # Execute ELAN
    if platform.system() == "Linux":
      elan_proc = subprocess.Popen([self.elan_exec, eaf_path])
    elif platform.system == "Darwin":
      elan_proc = subprocess.Popen([self.elan_exec, eaf_path])
    elif platform.system == "Windows":
      elan_proc = subprocess.Popen([self.elan_exec, eaf_path])



if __name__=='__main__':
  gui = ElanGui()
