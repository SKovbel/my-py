# https://medium.com/dida-machine-learning/how-to-recognise-objects-in-videos-with-pytorch-44f39f4c22f9
import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""
import cv2
import pafy
import matplotlib.pyplot as plt

import torch
from sample_class import ObjectDetectionPipeline

path = lambda name: os.path.join(os.path.join(os.path.dirname(__file__), f"../../tmp/pytorch/cv2"), name)
os.makedirs(path(''), exist_ok=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
url = "https://www.youtube.com/watch?v=wqctLW0Hb_0"

def get_youtube_cap(url):
    play = pafy.new(url).streams[-1] # we will take the lowest quality stream
    assert play is not None # makes sure we get an error if the video failed to load
    return cv2.VideoCapture(play.url)


print('download ...')
cap = get_youtube_cap("https://www.youtube.com/watch?v=usf5nltlu1E")
print('downloaded')
ret, frame = cap.read()
print('read...')
cap.release()
print('release...')
plt.imshow(frame[:,:,::-1]) # OpenCV uses BGR, whereas matplotlib uses RGB
#plt.show()


print('obj_detect...')
obj_detect = ObjectDetectionPipeline(device="cpu", threshold=0.5)
print('obj_detected')

plt.figure(figsize=(10,10))
plt.imshow(obj_detect(frame)[:,:,::-1])
plt.show()



batch_size = 16

cap = get_youtube_cap(url)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

size = min([width, height])

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(path("out.avi"), fourcc, 20, (size, size))

obj_detect = ObjectDetectionPipeline(device="cuda", threshold=0.5)
print('loop...')

exit_flag = True
while exit_flag:
    print('.', end='')
    batch_inputs = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if ret:
            batch_inputs.append(frame)
        else:
            exit_flag = False
            break

    outputs = obj_detect(batch_inputs)
    if outputs is not None:
        for output in outputs:
            out.write(output)
    else:
        exit_flag = False

cap.release()