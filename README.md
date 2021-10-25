# Project Road Segmentation

In this repo, we used one U-Net model to complete the road segmentation classification task which is the project 2 in CS-433 class at EPFL. The final result produced by the best model has F1-Score of 0.90 on AIcrowd competition evaluation system 

## Requirement
* numpy==1.18.5
* scikit_image==0.16.2
* torchvision==0.8.1+cu101
* pandas==1.1.5
* imutils==0.5.3
* matplotlib==3.2.2
* torch==1.7.0+cu101
* opencv_python==4.1.2.30
* skimage==0.0

## Run
Since the mian code of U-Net is run on Google Colab, and our computers do not have any CUDA, we added several run.py for using as you prefer.
And the Colab notebook link for U-Net is: https://drive.google.com/drive/folders/1YpBBNno5EF80QGGEJbc-tsSYP67HfQxu?usp=sharing (the U_net.ipynb is the main notebook). 

##### All our run files need under CUDA environment !!! Each training for best U-Net model is about 3 hours

* run.py: this is the run file for training model and generating submission.csv used on Colab 
(in Colab I used this command: !python3 '/content/drive/My Drive/Colab/ML_road_segment_classfication/scratch_UNet/run.py')

* run_loadmodel.py: this is the run file loading pre-trained model parameters and generating submission.csv. Also this is used on Colab
(in Colab I used this command: !python3 '/content/drive/My Drive/Colab/ML_road_segment_classfication/scratch_UNet/run_loadmodel.py')

* run_local.py: this is the run file for training model and generating submission.csv used locally
(please use command: python3 run_local.py)

* run_loadmodel_local.py: this is the run file loading pre-trained model parameters and generating submission.csv used locally.
(please use command: python3 run_loadmodel_local.py)

##### PS: 
Since our pre-trained model is based on CUDA, the loading function need CUDA too. However, due to the computer limitation, we could not test the run_local.py and run_loadmodel_local.py locally. So we are not sure about whether these two run files could run correctly or not. If you find these two files have some errors, please contack me: lei.pang@epfl.ch or jingran.su@epfl.ch

But for first two run files running on Colab, they are valid but it seems could only run in my own drive.
If you want to run these two files on Colab, please upload all files included in that Colab link (notebook and .py etc) in your own Google drive. And direct the notebook to your own drive using following two commands:

from google.colab import drive

drive.mount('/content/drive')

and then you could use the two Colab command in your own Colab notebook to run the run.py and run_loadmodel.py files.

Thanks for you understanding.



## git structure
<pre>
.

├── README.md                            
├── training            
│   └── images                           Satelite images (400x400)
|   └── groundtruth
├── test_set_images                      test images (608x608)
│
├── scratch_UNet                         U-Net model
│   └── pics                             figures used in report
│   └── run.py                           used on Colab to train model
│   └── run_loadmodel.py                 used on Colab to load trained model
│   └── run_local.py                     used locally to train model
│   └── run_loadmodel_local.py           used locally to load trained model
│   └── U_net                            notebook for all procedures
│   └── Feature_Augmentation.py          
│   └── get_patch.py
│   └── help_functions.py
│   └── load_data.py
│   └── Model_U_net.py                   model architechture
│   └── Prepare_process.py
│   └── requirement.txt                  requirement libraries and corresponding version
│   └── Submission.py
│   └── Tensor_dataset.py                to generate tenso dataset
│   └── Testing_process.py
│   └── Training_process.py
│   └── U_Net_parameter.pt               pre-trained model parameters used in run_loadmodel.py
│   └── Validation_functions.py
├── CNN                                  CNN model
│   └── CNN.ipynb                        model architechture
│   └── cnn.h5                           a saved weight of CNN
├── RFC                                  Random forest classifier
│   └── helper.py                        
│   └── submission.py                    
│   └── RFC_run.py                       run file of RFC model

