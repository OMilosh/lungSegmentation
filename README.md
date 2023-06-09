# lungSegmentation
Code for lung segmentation. Training with Unet on the Montgomery County X-ray Set and Shenzhen datasets.
Download datasets from https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels.
```bash
Project structure:
|   .gitignore
|   dataloader_func.py
|   preprocessing.py
|   README.md
|   requirements.txt
|   train.py
|   val_func.py
|     
+---archive
|   \---Lung Segmentation
|       |   NLM-ChinaCXRSet-ReadMe.docx
|       |   NLM-MontgomeryCXRSet-ReadMe.pdf
|       |   
|       +---.ipynb_checkpoints
|       |       Montgomery-checkpoint.ipynb
|       |       
|       +---ClinicalReadings
|
+---index_split
|       test_index.csv
|       train_index.csv
|       val_index.csv
```
## Loss plot:
![image](https://github.com/OMilosh/lungSegmentation/assets/83598973/e229be8f-7836-491e-b9a6-8b72d77d615c)
## U-net result:
![image](https://github.com/OMilosh/lungSegmentation/assets/83598973/b972b941-4c57-480e-836d-2dcda9b26e37)
## Final CXR: post-processing includes finding the contours of the lungs to correct "holes" and "extra details" in the resulting masks. 
![image](https://github.com/OMilosh/lungSegmentation/assets/83598973/da038967-e892-4f1f-a099-e525f70e56f2)
