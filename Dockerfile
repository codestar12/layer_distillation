FROM tensorflow/tensorflow:2.0.0-gpu-py3
RUN mkdir -p /home/cody/dataset/
RUN mkdir -p /home/cody/layer_distillation/
WORKDIR /home/cody/layer_distillation/

CMD ["python", "iterative_replacement-imagenet.py"]
