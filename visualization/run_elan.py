from create_elan import *
import tkinter as tk

import os
import platform 
import subprocess


class ElanGui:

  def __init__(self, video_filepath=""):
    self.window = tk.Tk()
    self.window.geometry("800x400")


    elan_exec_text = tk.StringVar()
    elan_exec_text.set("Path to ELAN executable")
    elan_exec_label = tk.Label(self.window, textvariable=elan_exec_text, height=2)
    elan_exec_label.grid(row=1, column=1)
    self.elan_exec_textbox = tk.Text(self.window, height=2, width=40)
    self.elan_exec_textbox.insert("end", "/home/ishan/Documents/research/ccg/elan/bin/ELAN")
    self.elan_exec_textbox.grid(row=1, column=2)


    mlf_fp_text = tk.StringVar()
    mlf_fp_text.set("MLF filepath name for annotations")
    mlf_fp_label = tk.Label(self.window, textvariable=mlf_fp_text, height=2)
    mlf_fp_label.grid(row=2, column=1)
    self.mlf_textbox = tk.Text(self.window, height=2, width=40)
    self.mlf_textbox.insert("end", "/home/ishan/Documents/research/ccg/copycat/CopyCat-HTK/projects/MediaPipe/results/res_hmm5.mlf")
    self.mlf_textbox.grid(row=2, column=2)


    video_dir_text = tk.StringVar()
    video_dir_text.set("Video directory name")
    video_dir_label = tk.Label(self.window, textvariable=video_dir_text, height=2)
    video_dir_label.grid(row=3, column=1)
    self.video_dir_textbox = tk.Text(self.window, height=2, width=40)
    self.video_dir_textbox.insert("end", "/home/ishan/Documents/research/ccg/copycat/DATA/input")
    self.video_dir_textbox.grid(row=3, column=2)


    eaf_savedir_text = tk.StringVar()
    eaf_savedir_text.set("Elan annotation file save directory")
    eaf_savedir_label = tk.Label(self.window, textvariable=eaf_savedir_text, height=2)
    eaf_savedir_label.grid(row=4, column=1)
    self.eaf_savedir_textbox = tk.Text(self.window, height=2, width=40)
    self.eaf_savedir_textbox.insert("end", "/home/ishan/Documents/research/ccg/elan")
    self.eaf_savedir_textbox.grid(row=4, column=2)

    self.next_video_button = tk.Button(self.window, text="Start Video Loop", command=self.launch_elan)
    self.next_video_button.grid(row=5, column=2)

    self.close_button = tk.Button(self.window, text="Quit", command=self.window.destroy)
    self.close_button.grid(row=6, column=2)


    self.window.mainloop()


  def launch_elan(self):

    # Initialize next video button, read textboxes after that
    self.is_elan_launched = True
    next_video_button_clicked = tk.IntVar()
    self.next_video_button.configure(text="Next Video")

    # Get path to ELAN executable
    elan_exec = self.elan_exec_textbox.get("1.0", "end-1c")
    self.elan_exec_textbox.pack_forget()

    # Get MLF filepath
    mlf_fp = self.mlf_textbox.get("1.0", "end-1c")
    annotations = mlf_to_dict(mlf_fp)
    self.mlf_textbox.pack_forget()

    # Get all videos to iterate over. TODO: Have options for selecting specific videos or something
    video_dir = self.video_dir_textbox.get("1.0", "end-1c")
    video_fps = []
    for root, dirs, files in os.walk(video_dir):
      for video_fp in files:
        if video_fp.endswith(".mp4"):
          video_fps.append(os.path.join(root, video_fp))
    self.video_dir_textbox.pack_forget()

    # Get EAF savedir
    eaf_savedir = self.eaf_savedir_textbox.get("1.0", "end-1c")
    self.eaf_savedir_textbox.pack_forget()

    # Iterate through videos
    for video_fp in video_fps: 
      video_len = int(float(FFProbe(video_fp).video[0].duration) * 1000)
      annotations = scale_annotations(annotations, video_len)

      eaf_obj = make_elan(annotations, has_states=True, video_dirs=[video_fp], \
        eaf_savedir=eaf_savedir)[0]
      
      eaf_path = os.path.join(eaf_savedir, os.path.basename(os.path.splitext(video_fp)[0])) + ".eaf"
      if platform.system() == "Linux":
        os.system(elan_exec + " " + eaf_path)
        # subprocess.call(["xdg-open", eaf_path])
      elif platform.system == "Darwin":
        # TODO: use open
        pass
      elif platform.system == "Windows":
        os.startfile(eaf_path)

      # Wait for button to be clicked again before proceeding to next video
      self.next_video_button.wait_variable(next_video_button_clicked)
      next_video_button_clicked.set(0)




if __name__=='__main__':
  gui = ElanGui()
