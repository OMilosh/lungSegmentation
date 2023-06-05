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
![image](https://github.com/OMilosh/lungSegmentation/assets/83598973/41875f63-d824-4140-8676-b820ada71fcc)
