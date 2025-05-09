import pyaudio
import numpy as np
import keyboard
import cv2
from scipy.signal import lfilter, bilinear
from pyk4a import PyK4A
from pyk4a.config import Config, ColorResolution
from pick_test import *

# ===================================================
#               1) Global Configuration
# ===================================================
FORMAT = pyaudio.paInt16
CHANNELS = 7  # Azure Kinect has 7-mic array
RATE = 44100  # Sampling rate
CHUNK = 4096  # Buffer size
REFERENCE_PRESSURE = 2e-5  # 20 micropascals (standard reference pressure)
CALIBRATION_SENSITIVITY = 10  # Adjusted for real calibration
ALERT_THRESHOLD = 85  # dB SPL threshold for warnings
TARGET_CHANNEL = 0  # Select mic channel to measure
BACKGROUND_OFFSET = 0
NOISE_THRESHOLD = 0.003

# ===================================================
#  A-Weighting Filter Design (IIR based)
# ===================================================
def a_weighting(fs):
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997
    nums = [(2 * np.pi * f4) ** 2 * 10 ** (A1000 / 20.0), 0, 0, 0, 0]
    den1 = [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2]
    den2 = [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2]
    den3 = [1, 2 * np.pi * f3]
    den4 = [1, 2 * np.pi * f2]
    dens = np.polymul(np.polymul(den1, den2), np.polymul(den3, den4))
    b, a = bilinear(nums, dens, fs)
    return b, a

# ===================================================
#     2) SPL Calculation Function (WITH A-WEIGHTING)
# ===================================================
def calculate_spl(audio_data, calibration_offset, background_offset):
    float_data = audio_data.astype(np.float32) / 32768.0
    float_data *= 0.632  # Kinect mic sensitivity
    rms_value = np.sqrt(np.mean(float_data ** 2))
    rms_value = max(rms_value, 1e-15)
    spl = 20.0 * np.log10(rms_value / REFERENCE_PRESSURE) + calibration_offset
    if rms_value < NOISE_THRESHOLD:
        spl += background_offset
    return spl

# ===================================================
#     3) Overlay Display Function
# ===================================================
def show_overlay_image(image, spl_value, alert_threshold):
    text_color = (0, 255, 0)
    if spl_value >= alert_threshold:
        text = f"WARNING! SPL: {spl_value:.2f} dBA"
        text_color = (0, 0, 255)
        image = apply_orange_overlay(image,intensity=0.4)
    else:
        text = f"SPL: {spl_value:.2f} dBA"

    cv2.putText(image, text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)
    return image

# ===================================================
#     4) Main Execution
# ===================================================
if __name__ == "__main__":
    audio = pyaudio.PyAudio()
    device_index = None
    print("Searching for Azure Kinect microphone...")
    audio.get_device_count()
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if "Kinect" in info["name"]:
            device_index = i
            print(f"Azure Kinect mic found at index {i}: {info['name']}")
            break

    if device_index is None:
        print("Error: Azure Kinect microphone not found.")
        audio.terminate()
        exit(1)

    b_a, a_a = a_weighting(RATE)
    k4a = PyK4A(Config(color_resolution=ColorResolution.RES_720P))
    k4a.start()

    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=CHUNK)
        print(f"Measuring noise levels at fs={RATE} Hz with A-weighting filter...")
        print("Press 'q' to stop.")

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
            target_audio_data = audio_data[:, TARGET_CHANNEL]
            weighted_audio_data = lfilter(b_a, a_a, target_audio_data)
            spl_value = calculate_spl(weighted_audio_data, CALIBRATION_SENSITIVITY, BACKGROUND_OFFSET)

            capture = k4a.get_capture()
            if capture.color is not None:
                frame = capture.color
                frame = cv2.resize(frame, (960, 540))
                frame = show_overlay_image(frame, spl_value, ALERT_THRESHOLD)
                cv2.imshow("Azure Kinect SPL Monitor", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        k4a.stop()
        cv2.destroyAllWindows()
        print("Audio and video streams stopped.")