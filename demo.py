from PIL import Image

import cv2
import numpy as np
import snook.data as sd
import snook.model as sm
import sys
import torch


COLORS = [(0, 0, 0), (255, 255, 255), (0, 255, 255), (0, 0, 255), (255, 0, 0)]

autoencoder = "notebooks/models/autoencoder_ood.ts"
classifier = "notebooks/models/classifier_ood.ts"

snook = sm.Snook(autoencoder, classifier)
snook = snook.eval().cuda()

resize = sd.ResizeSquare(512)

path = sys.argv[1]
cap = cv2.VideoCapture(path)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = resize(frame)
    frame = np.array(frame) / 255.0
    
    with torch.no_grad():
        frame = torch.Tensor(frame).permute((2, 0, 1)).cuda()
        _, peaks, labels = snook(frame)

    peaks = peaks.cpu().numpy()
    labels = labels.cpu().numpy()
    frame = frame.permute((1, 2, 0)).cpu().numpy()
    
    frame = (frame * 255.0).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    for pos, label in zip(peaks, labels):
        cv2.circle(frame, tuple(pos)[::-1], 12, COLORS[label], 2)

    cv2.imshow("Snook", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()