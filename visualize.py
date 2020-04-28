import sys
sys.path.append('.')

import numpy as np
import argparse
import matplotlib.pyplot as plt

from Animation import Animation
from InputFrame import InputFrame
from OutputFrame import OutputFrame
from GatingFrame import GatingFrame

from utils import *
from matplotlib import widgets

np.set_printoptions(precision = 3, suppress = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data-path", type = str, default = "./data/test32.npy", help = "Path of the dataset")
    
    args = parser.parse_args()

    # Loading
    data = np.load(args.data_path)
    print ("Loaded")
    
    input_data = data[:,:5437]
    output_data = data[:,5437:]
    phase_data = np.array([get_phase(x)[7] for x in input_data]) # Get the phase info
    

    paused = False
    def set_start(text):
        try:
            anim.select_clip(int(text))
        except:
            print("Invalid clip number!")
            textbox.set_val(anim.clip_idx)
    def set_len(text):
        try:
            anim.set_clip_len(int(text))
        except:
            print("Invalid length!")
            textbox.set_val(anim.clip_len)
    def pause(event):
        global paused
        paused = not paused
        if paused:
            anim.pause()
            pausebutton.label.set_text("$\u25B6$")
        else:
            anim.resume()
            pausebutton.label.set_text("$\u25A0$")
    def next(event):
        anim.next()
        textbox.set_val(str(anim.clip_idx))
    def prev(event):
        anim.prev()
        textbox.set_val(str(anim.clip_idx))
    def on_slider(val):
        if val != anim.frame_val:
            anim.frame_idx = int(val * anim.clip_len)
    def print_clip(val):
        data = anim.data[anim.clip_idx * anim.clip_len: anim.clip_idx * anim.clip_len + anim.clip_len]
        print (data)
        np.savetxt("out.txt", data)

    fig = plt.figure("Controls", figsize = (3,3,))
    textbox =  widgets.TextBox(plt.axes([0.4,0.8,0.4,0.1]), "Clip:", '0')
    lentextbox =  widgets.TextBox(plt.axes([0.4,0.7,0.4,0.1]), "Clip Length:", '240')

    prevbutton = widgets.Button(plt.axes([0.25, 0.4, 0.1,0.1]), "$\u29CF$")
    pausebutton = widgets.Button(plt.axes([0.25 + 0.2, 0.4, 0.1,0.1]), "$\u25A0$")
    nextbutton = widgets.Button(plt.axes([0.25 + 0.4, 0.4, 0.1,0.1]), "$\u29D0$")
    widgets.TextBox(plt.axes([0.1,0.2,0.8,0.1]),  "", "Total number of clips: %d " % (len(data) / 240))
    frameslider = widgets.Slider(plt.axes([0.2, 0.55, 0.6, 0.06]), "", 0 , 1, 0)
    printbutton = widgets.Button(plt.axes([0.25, 0.08, 0.2,0.1]), "Print")
    
    # Drawing
    anim = Animation(data, frameslider)
    anim.select_clip(0)


    prevbutton.on_clicked(prev)
    pausebutton.on_clicked(pause)
    nextbutton.on_clicked(next)
    textbox.on_submit(set_start)
    lentextbox.on_submit(set_len)
    frameslider.on_changed(on_slider)
    printbutton.on_clicked(print_clip)
    anim.draw(130)
    anim.play()
    plt.show()