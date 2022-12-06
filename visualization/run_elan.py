from create_elan import *
import tkinter as tk

import os


class ElanGui:

  def __init__(self, video_filepath=""):
    self.window = tk.Tk()
    self.video_filepath = ""

  def launch_elan(self, launch_executable):
    os.system()

  def launch_gui(self, options):
    # plan out gui
    # button for output current alignment
    # button for next video?
    if executable_file_name not in options:
      pass # check textbox, do this for other args to funcs too

    video_name = self.video_filepath.split('/')[-1]

    launch_elan_button(self.window, command=lambda _:self.launch_elan(executable_file_name))
    output_alignment_button = tk.Button(self.window, command=lambda _:self.output_alignment(video_name))
    self.window.mainloop()

