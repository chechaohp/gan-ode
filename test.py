import torch
from on_dev.mocogan_ode import VideoGenerator

gen = VideoGenerator(1,50,0,16,16,ngf=28)
out = gen.sample_videos(1)[0]
print(out.size())