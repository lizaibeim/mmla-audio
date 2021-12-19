# User Manual

## Create the virtual environment and install the required third-party libraries

### 1. In terminal and under the current project directory, type the following commands:

```
pip3 install virtualenv
```
```
python3 -m virtualenv venv
```

This will create a virtual environment directory *.\venv\\*

----

### 2. Activate the virutal environment by typing the following commands:

#### For Windows
```
cd ./venv/Scripts/
```
This will re-direct the current working directory to the Scripts folder. Then, activate the bash *activate.bat* bash file by typing in
```
./activate
```

#### For Mac/Linux
```
cd ./venv/bin/
```

```
source activate
```

---


### 3. Run the *setup.py* by typing the following commands
```
python3 ./../../setup.py
```

The *setup.py* will automatically install the required libraries. For macosx user, after running the setup script, you need to install the pyAudio library manually by typing the following commands:

```
brew install portaudio
```

```
pip3 install pyAudio
```

### 4. After successfully installing the rquired libraries, you need to add a new python intepreter if you are using pycharm by using the existing virtual environment. You can run it on terminal without setting the python intepreter.

Make sure you have entered the virtual environment so that the *(venv)* will show at the very begining of the prompt in terminal.

#### Go to *./OverlapDection* or *./SpeakerIdentification* folder for running the overlap detector or speaker identification model. Follow the prompts on console to start.

```
python3 ./record_on_pc.py
```
