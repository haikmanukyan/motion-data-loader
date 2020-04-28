import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from InputFrame import InputFrame
from OutputFrame import OutputFrame
from utils import get_phase
from InputFrame import InputFrame
from OutputFrame import OutputFrame
from GatingFrame import GatingFrame

class Animation:
    def __init__(self, data, frameslider = None, clip_len = 240):
        self.data = data
        self.frameslider = frameslider
        self.animations = []
        self.titles = []
        self.axes = []
        self.clip_idx = 0
        self.frame_idx = 0
        self.is_paused = False
        self.set_clip_len(clip_len)

    def set_clip_len(self, clip_len):
        self.clip_len = clip_len
        self.n_clips = int(len(self.data) / self.clip_len)
        self.select_clip(self.clip_idx)

    def next(self, event = None):
        self.clip_idx += 1
        self.clip_idx = self.clip_idx % self.n_clips

        self.select_clip(self.clip_idx)

    def prev(self, event = None):
        self.clip_idx -= 1
        self.clip_idx = self.clip_idx % self.n_clips
        self.select_clip(self.clip_idx)

    def select_clip(self, clip_idx):
        self.clip_idx = clip_idx
        data = self.data[self.clip_idx * self.clip_len: self.clip_idx * self.clip_len + self.clip_len]
        input_data = data[:,:5437]
        output_data = data[:,5437:]
        phase_data = np.array([get_phase(x)[7] for x in input_data]) # Get the phase info

        self.titles = ["Input", "Output", "Phase"]
        self.animations = [
            self.get_animation(output_data, OutputFrame),
            self.get_animation(input_data, InputFrame),
            self.get_animation(phase_data, GatingFrame)
        ]
    
    def get_animation(self, data, frame_type):
        frames = [frame_type(x) for x in data]
        return frames

    def add_axis(self, fig, idx, frame, title = None):
        if frame.projection == "3d":
            ax = fig.add_subplot(idx, projection="3d")
            size = 1.2
            ax.set_xlim3d(-size,size)
            ax.set_ylim3d(-size,size)
            ax.set_zlim3d(0,2 * size)

        else:
            ax = fig.add_subplot(idx)
            ax.axis('equal')
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)

        # ax.axis('off')
        # ax.set_title(title)

        graph = frame.draw(ax)
        return graph

    def draw(self, format = 220):
        h = int(format / 100) * 4
        w = int((format % 100) / 10) * 4
        self.fig = plt.figure(figsize = (12,4))
        self.fig.tight_layout()
        self.graphs = []

        for i, (animation, title) in enumerate(zip(self.animations, self.titles)):
            self.graphs.append(self.add_axis(self.fig, format + 1 + i, animation[0], title))

    def update(self, frame_idx):
        if not self.is_paused:
            self.frame_idx = (self.frame_idx + 1) % self.clip_len
        if self.frameslider != None:
            self.frame_val = self.frame_idx / self.clip_len
            self.frameslider.set_val(self.frame_val)
            self.frameslider.label.set_text(self.frame_idx)
        ret = []
        for animation, graph in zip(self.animations, self.graphs):
            ret += animation[self.frame_idx].update(graph)
        return ret

    def play(self):
        self.anim = anim.FuncAnimation(self.fig, self.update, len(self.animations[0]), interval = 1000 / 120., blit=True)

    def pause(self):
        self.is_paused = True
    def resume(self):
        self.is_paused = False

    def save(self, path = 'data/animation.gif'):
        self.anim.save(path, writer = 'imagemagick', fps = 30)