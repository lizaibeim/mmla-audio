import os


Root_Dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

"""path creation"""
paths = [os.path.join(Root_Dir, 'OverlapDetection/experiment/charts/'),
         os.path.join(Root_Dir, 'OverlapDetection/experiment/logs/'),
         os.path.join(Root_Dir, 'OverlapDetection/experiment/recordings/'),
         os.path.join(Root_Dir, 'OverlapDetection/experiment/recordings/post-time/'),
         os.path.join(Root_Dir, 'OverlapDetection/experiment/recordings/post-time/whole/'),
         os.path.join(Root_Dir, 'OverlapDetection/experiment/recordings/post-time/features/'),
         os.path.join(Root_Dir, 'OverlapDetection/experiment/recordings/post-time/segments/'),
         os.path.join(Root_Dir, 'OverlapDetection/experiment/recordings/post-time/standardized/'),
         os.path.join(Root_Dir, 'OverlapDetection/experiment/recordings/real-time/'),
         os.path.join(Root_Dir, 'SpeakerIdentification/experiment/charts/'),
         os.path.join(Root_Dir, 'SpeakerIdentification/experiment/logs/'),
         os.path.join(Root_Dir, 'SpeakerIdentification/experiment/recordings/'),
         os.path.join(Root_Dir, 'SpeakerIdentification/experiment/recordings/post-time/'),
         os.path.join(Root_Dir, 'SpeakerIdentification/experiment/recordings/post-time/whole/'),
         os.path.join(Root_Dir, 'SpeakerIdentification/experiment/recordings/post-time/segments/'),
         os.path.join(Root_Dir, 'SpeakerIdentification/experiment/recordings/post-time/standardized/'),
         os.path.join(Root_Dir, 'SpeakerIdentification/experiment/recordings/real-time/')
         ]

for path in paths:
    if not os.path.exists(path):
        print('Create path: ', path)
        os.mkdir(path)


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
