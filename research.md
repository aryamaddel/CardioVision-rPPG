# CardioVision-rPPG

Remote Photoplethysmography (rPPG) pipeline for non-contact heart rate estimation from facial video. Uses POS algorithm (Wang et al. 2017) with MediaPipe face tracking.

```
python main.py                              # webcam
python main.py --source video.mp4           # video file
python main.py --source 1 --window 8        # camera 1, 8s window
```

Left window: face ROI + BPM. Right window: Eulerian Video Magnification.
