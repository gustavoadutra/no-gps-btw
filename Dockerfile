FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime


RUN pip install opencv-python imageio matplotlib pillow torchsummary

RUN apt update -y && apt install -y libgl1-mesa-glx libglib2.0-0