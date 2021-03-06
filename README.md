# phone_finder
Repository contains a dataset of 100+ images that has a phone. The goal was to use any algorithm to localize the phone in the images. There are several scripts that are used to localize the phone in the picture. This was a coding challenge I was given in one of my job interviews.

![download (1)](https://user-images.githubusercontent.com/85404022/126579907-db8c15ea-49c5-43fd-a7fa-ceafff4596d3.png)

## Folder & file info
*********

find_phone - the folder containing all the training images

find_phone_test_images - I just created this folder to put some images and test my functions.

Bayesian Model.ipynb - This is a jupyter notbook that I created explaning my approach to solve the problem. 
		 	I tested several deep learning methods but I failed to get a good accuracy. So I used
			a basic Bayesian method to estimate the coordinates. This notebook explains the model
			and the steps i took to estimate the coordinates.

Find phone.pdf - Problem statement

find_phone.py - The .py file to test an image. Please note that in the command prompt you will have to type
		"directory > python find_phone.py /find_phone_test_images/51.jpg" for the file to run. The 
		"/find_phone_test_images/51.jpg" is identified as a string and it will give an error. 


Deep Learning.ipynb - A jupyter notebook that I have tested several deep learning
		methods to localize the phone. The algorithms used are convolutional neural network, a residual network and a pretrained
    		algorithm (mobile net v2). 

Image Segmentation - A jupyter notebook where I have used Unet algorithm to segment the phone from the image. The masks were created
manually, so the segmentation is not 100% clear. However, given the mask the image segmentation algorithm is doing a good job. 

train_phone_finder.py - This is the train_phone_finder file that trains the model. Similarly like the previous .py
			file you will have to type "/find_phone" instead of "~/find_phone" to locate the folder.


trained_prior.npy - This is the trained model using the images in the find_phone folder. If you run the training script
			the model will over write this file. 	

## Few images from the notebook

**A detected and not detected image fromt the Bayesian model**

![download](https://user-images.githubusercontent.com/85404022/126579617-50d568ce-168a-4db9-bb38-251456bc0746.png)

![download](https://user-images.githubusercontent.com/85404022/126579780-0a3b824e-626c-448a-a7af-5995b4991f2b.png)

**A figure from the image segementation. The original image, the created mask and the predicted mask by the Unet algorithm is shown.**

Note that the mask was created manually by masking about the center of the phone.

![download (2)](https://user-images.githubusercontent.com/85404022/126580232-0db14c8a-6db6-4c8a-b1eb-b471852317aa.png)

![download (3)](https://user-images.githubusercontent.com/85404022/126580240-6a37cbb7-ccf7-4931-af40-69d59b30ec68.png)


