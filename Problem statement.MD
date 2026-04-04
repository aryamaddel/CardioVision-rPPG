HC02 - Contact-Free Cardiac Stress Monitoring via Facial Video Analysis (rPPG)
Domain- Healthcare & Accessibility
THE PROBLEM
Monitoring heart rate and cardiac stress requires wearable hardware — ECG patches, pulse oximeters,
smartwatches — that are unavailable in disaster triage zones, remote telehealth scenarios, and lowincome clinical settings. Yet the physiological signal of a heartbeat is already present in any standard
smartphone camera: microscopic green-channel intensity fluctuations in facial skin caused by blood
pulsing through capillaries. If this signal can be reliably extracted, every smartphone becomes a zerofriction biometric monitor, enabling cardiac triage without any additional hardware. The challenge is that
this 'photoplethysmographic' signal is orders of magnitude smaller than the noise introduced by ambient
lighting, subject movement, and video compression artifacts.
CORE TECHNICAL COMPLEXITY
Remote Photoplethysmography (rPPG) requires isolating a pulse signal from a video stream by
analyzing per-frame green-channel intensity variations in a facial Region of Interest (ROI). The core
difficulty is signal separation: teams must implement advanced noise rejection — such as Independent
Component Analysis (ICA), the Plane-Orthogonal-to-Skin (POS) algorithm, or a deep rPPG neural
network — to filter out ambient lighting flicker, subject head motion, and codec compression noise. The
extracted Inter-Beat Interval (IBI) variability must then feed a stress classifier. The system must also
include a signal quality indicator: when lighting or motion conditions degrade the signal below a
confidence threshold, it must refuse to output a reading rather than silently returning an inaccurate one.
EXPECTED OUTCOME
A system that processes a 30-second selfie video from a standard smartphone camera and outputs:
estimated Heart Rate in BPM, Inter-Beat Interval (IBI) variability as an autonomic stress proxy, a binary
or graded stress classification, and a signal quality confidence score — refusing output when
confidence is too low to be clinically meaningful.
12
EXACT DELIVERABLES
• A video processing pipeline that detects a facial ROI and extracts per-frame green-channel
intensity signals.
• A noise rejection module implementing at least one advanced technique: ICA, POS algorithm, or
a pre-trained deep rPPG model.
• A stress classifier trained on IBI variability features (RMSSD, SDNN, or LF/HF ratio).
• A signal quality confidence indicator that flags and suppresses unreliable readings caused by
poor lighting or excessive motion.
• A live demo UI processing a 30-second test video and displaying BPM, IBI variability graph, and
stress classification.
RECOMMENDED TECH APPROACH
Use a computer vision library for facial detection and ROI tracking. Use a signal processing library for
bandpass filtering, FFT, and ICA. Optionally use a pre-trained deep rPPG model for the noise rejection
step. Use an open rPPG benchmark dataset for validation (e.g., UBFC-rPPG or PURE).
TWIST 1
rPPG is unreliable when the patient is moving or lighting is poor. Rather than failing silently, the system
must build a Triage Decision Agent: when the signal-to-noise ratio of the rPPG feed drops below a
confidence threshold, the pipeline automatically switches to a Visual Assessment Mode — using a
lightweight CNN or Vision-Language Model to analyze visible physical signs of distress such as pallor,
sweating, or labored breathing, and output a Visual Stress Score as a fallback. The dashboard must
display which mode is currently active and why, toggling dynamically between Biometric Mode and
Visual Assessment Mode based on whichever signal the agent deems more trustworthy in the current
environment.
 TWIST 2
Real-time inference is disabled due to device overheating. The system must fall back to accepting a
pre-recorded 10-second MP4 video upload, processing it fully in the backend to extract a static average
BPM — maintaining accuracy without any live tracking.