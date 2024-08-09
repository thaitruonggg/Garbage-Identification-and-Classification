# Garbage Identification and Classification
## LINK FOR KAGGLE NOTEBOOK TO TRAIN MODELS:
- ResNet50: https://www.kaggle.com/code/naofunyannn/sic-resnet50
- Swin Transformer:https://www.kaggle.com/code/naofunyannn/sic-swintransformer
## PYTHON PACKAGES REQUIREMENT:
- PyTorch (CUDA version is recommended)
- Matplot
- Numpy
- OpenCV
## INSTRUCTION
### If you train on your device, please start from here:
1. Due to Github limitation, please download all the classes folders with the link below, then unzip, copy and paste it into the `all_classes` folder (your directory should be ***all_classes/classes - Ex: all_classes/battery***). Link: https://www.kaggle.com/datasets/naofunyannn/sicdataset/data
2. Run `Training_ResNet.py` or `Training_SwinTransformer.py`.
3. Check `models` folder to see if `ResnetModel.pth` or `SwinTransformerModel.pth` available or not -> Then continue to step 5
### If you train your model using Kaggle, please continue from here:
4. Copy your downloaded model into the `models` folders
5. Run `Execute_Images_ResNet.py` or `Execute_Images_SwinTransformer.py` to identify and classify garbage in images (Using images / videos in `external` folder).
6. Run `Execute_Videos_ResNet.py` or `Execute_Videos_SwinTransformer.py` to identify and classify garbage in videos or real-time webcam (Using images / videos in `external` folder).
## RESULT
### Images
![Screenshot 2024-07-17 093230](https://github.com/user-attachments/assets/e702785d-3a18-4050-9b86-00753e3a0e30)
### Videos
![Screenshot 2024-07-17 093637](https://github.com/user-attachments/assets/91760098-6e9b-46a7-996e-5a6bd5093265)

