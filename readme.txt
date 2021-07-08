folder & files info
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


testing.ipynb - A jupyter notebook that I was testing some algorithms on. I have tested several deep learning
		methods. I also used this as a scrap notebook to test things out.

train_phone_finder.py - This is the train_phone_finder file that trains the model. Similarly like the previous .py
			file you will have to type "/find_phone" instead of "~/find_phone" to locate the folder.


trained_prior.npy - This is the trained model using the images in the find_phone folder. If you run the training script
			the model will over write this file. 	 