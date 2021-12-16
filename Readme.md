# User Manual

## Create the virtual environment and pre-install the  third-party libraries

### 1. In terminal and under the current project directory, type the command

```
pip install virtualenv
```
```
py -m virtualenv venv
```

This will create a virtual environment directory *.\venv\\*

----

### 2. activate the virutal environment by typing in
```
cd .\venv\Scripts\
```
This will re-direct the current working directory to the Scripts folder. Then, activate the bash *activate.bat* bash file by typing in
```
.\activate
```
---
### 3. run the *setup.py* by typing the following commands
```
cd ..
```
```
cd ..
```
```
python setup.py

```
### 4. After successfully installing the rquired libraries, you need to add a new python intepreter if you are using pycharm by using the existing virtual environment. 
---
### 5. Run the *run_on_pc.py* script under the *./OverlapDection* and *./SpeakerIdentification* folder for running the overlap detector and speaker identification model and follow the prompts on console.

