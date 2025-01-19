### Create virtual environment and download dependencies to run project

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Data Pre-processing command
#### Enter data_processing directory
```bash
$ cd data_processing
```
####Toxic wav file converter

```bash 
$ python3 wav_converter.py <toxic_dataset_directory_path> <wav_directory_path> toxic
```
####Non-Toxic wav file Converter

```bash
$ python3 wav_converter.py <non_toxic_dataset_directory_path> <wav_directory_path> non-toxic

```

####Csv file generater with label

```bash
$ python3 csv_generator.py <data_set_directory> <csv_file_path>
```

### Train Model

#### Enter train-model directory
```bash
$ cd train-model
```

#### Install all lib from train-model/requires.txt file

```bash
$ pip install -r requires.txt
```

####Change the hard coded directory of train and test dataset 

following variables need to change 

```bash
dataSetDir // dataset diretory
file_path // dataset csv path
testDir // test dataset directory 
test_file // test dataset csv path
```

### Run ToxicTrain.py either from IDE or command
```bash
$ python3 ToxicTrain.py
```