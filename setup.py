import os

libs = ["pandas", "keyboard", "librosa", "noisereduce", "numpy==1.21", "requests", "soundfile", "tensorflow",
        "webrtcvad-wheels",
        "python_speech_features", "Adafruit_IO", "PyAudio-0.2.11-cp39-cp39-win_amd64.whl", "scipy", "sklearn",
        "pyecharts", "numba==0.53", "opencv-python", "scikit-image"]

print("Install Virtual Environment")

try:
    for lib in libs:
        os.system("pip install " + lib)

    print("Install Successfully")
except:
    print("Virtual Environment Set Up Fails")
