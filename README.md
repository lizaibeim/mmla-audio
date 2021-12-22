# User Manual

## I. Create the virtual environment and install the required third-party libraries

### 1. In terminal and under the current project directory */MMLA_Audio/*, type the following commands:

```
pip3 install virtualenv
```
```
python3 -m virtualenv venv
```

This will create a virtual environment directory *.\venv\\*


### 2. Activate the virutal environment by typing the following commands:

#### For Windows
```
./venv/Scripts/activate.bat
```

#### For Mac/Linux
```
source ./venv/bin/activate
```

### 3. Run the *setup.py* by typing the following commands
```
python3 setup.py
```

The *setup.py* will automatically install the required libraries. For ***Mac/Linux*** user, after running the setup script, you need to install the pyAudio library manually by typing the following commands:

```
brew install portaudio
```

```
pip3 install pyAudio
```

### 4. After successfully installing the rquired libraries, you need to add a new python intepreter if you are using pycharm by using the existing virtual environment. You can run it on terminal without setting the python intepreter.

Make sure you have entered the virtual environment so that the ***(venv)*** will show at the very begining of the prompt in terminal.

### 5. Go to *./OverlapDection* or *./SpeakerIdentification* folder and then type the following command to start the overlap detector or speaker identification model.


```
python3 OverlapDetection/scripts/record_on_pc.py
```
or
```
python3 SpeakerIdentification/scripts/record_on_pc.py
```

---
## II. Guidelines for running overlap detector
### 1. Record ambient noise
You need to keep quiet for 10 seconds in order to record the ambient noise for calibrating the background noise.
### 2. Start to detect overlap
It will consecutively detect the overlaps for each 2.5 seconds.
### 3. Stop detecting
Press ctrl + 'c' will exit the program.

---
## III. Guidelines for running speaker identification
### 1. Record ambient noise
You need to keep quiet for 10 seconds in order to record the ambient noise for calibrating the background noise.
### 2. Register the speakers you want to identify during the conversation
The speaker need to read a 1 minute paragraph for registering your voice in the database. 
### 3. Wait for training
The model will do training based on the registered speakers' corpus.
### 4. Start to identify the speaker
It will consecutively identify the speaker for each 1.5 seconds.
### 5. Stop identifying
Press ctrl + 'c' will exit the program.


---
## IV. Analysing
### 1. Data
For all data, it would be stored under the ./experiment folder.
### 2. Diagram
You can draw the overlap degree and speaker time distribution digrams for visualizing the conversation by typing the following command:
```
python3 SpeakerIdentification/scripts/speaker_time_distribution.py
```
or
```
python3 OverlapDetection/scripts/overlap_degree_distribution.py
```

The corresponding pie charts is stored under the ./experiment/pie_charts/ directory.





