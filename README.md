# kaggle-cloud
This is an *improved* version of the code that was used during the competition which achieved      

*Public LB  : 55/1,538*

*Private LB : 26/1,538*

Inspired from [6th place solution](https://www.kaggle.com/c/understanding_cloud_organization/discussion/118017), I added pretraining-step using to make use of `train+test` set. 

## Installation
All requirements should be installed from requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n cloud python=3.6
source activate cloud
pip install -r requirements.txt
```

## Prepare dataset
```
$ kaggle competitions download -c understanding_cloud_organization
$ unzip understanding_cloud_organization.zip -d data
```
After downloading and unzipping, the data directory should look like this:
```
data
  +- train.csv
  +- sample_submission.csv
  +- train_images
  +- test_images
```

### Make Split
To make 5-fold stratified split, run
```
python make_splits.py
```
This should save `train_splits.csv` in data-folder

### Pre-Training
To pretrain using train and test set, run
```
python train.py --log_dir runs --model_name FPN_effb4 --pretrain True --num_epochs 5 --scheduler_name linear_warmup --data_dir ./data
```
This should save checkpoints in `./runs/FPN_effb4/fold_0/checkpoints`

### Main Training
The 2nd-stage training was initialized from the last checkpoint of pre-training step
```
python train.py --log_dir runs --model_name FPN_effb4 --fold 0 --num_epochs 30 --scheduler_name multistep --initial_ckpt ./runs/FPN_effb4/fold_0/checkpoints/4_train_loss.pth
```
To train with different `fold`, just change the fold number

### Take EMA
To take Exponential-moving-average of top-5 checkpoints, run
```
python swa.py --log_dir runs --model_name FPN_effb4 --fold 0
```
This should save a `0_model_ema_5.pth` checkpoint file in `./runs/FPN_effb4/fold_0/checkpoints` 

### Make Submission
To create a submssion-csv file using the best checkpoint file, run
```
python submit.py --initial_ckpt ./runs/FPN_effb4/fold_0/checkpoints/0_model_ema_5.pth --sub_name submission_fold0.csv
```
This should create a `submission_fold0.csv` file. To submit this to kaggle, run
```
kaggle competitions submit -c understanding_cloud_organization -f submission_fold0.csv -m "fold0_submission"
```
>Train 5-folds and enemble them to get better results on LB

