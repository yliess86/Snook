from PIL import Image

import argparse
import cv2
import numpy as np
import snook.data as sd
import snook.model as sm
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--autoencoder", "-a", type=str, default="notebooks/models/autoencoder_ood.ts", help="AutoEncoder TorchScript Path")
parser.add_argument("--classifier",  "-c", type=str, default="notebooks/models/classifier_ood.ts",  help="Classifier TorchScript Path")
parser.add_argument("--video",       "-v", type=str,                                                help="Video File Path")
args = parser.parse_args()

COLORS = [(0, 0, 0), (255, 255, 255), (0, 255, 255), (0, 0, 255), (255, 0, 0)]

snook = sm.Snook(args.autoencoder, args.classifier)
snook = snook.eval().cuda()

cap = cv2.VideoCapture(args.video)
resize = sd.ResizeSquare(512)

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