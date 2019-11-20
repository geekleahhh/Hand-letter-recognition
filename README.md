# Hand-letter-recognition
For VGG_features.py:
Here we use VGG features to feed into the classifiers instead of using 16*16 flattened images.

1. Firstly we generate 2 kinds of letters: "o" and "c".
2. Then store them and using VGG_features.py to generate features. The features will be 25088 dimension for each samples. All the features of "c" letters is stored in c.npy. All the features of "o" letters is stored in truec.npy.
3. After get 2 kinds of features of 2 letters, we transfrom them into txt format. When transfering, we need firstly to shuffle them. Then transform the float numbers into str.
4. At last, we get a features.txt which store features of 2 kinds of letters and feed them in to the classifiers of the last lab we get. Then we use it to train the classifer.
5. At last, same as before, we use the classifier to classify the letters in the camera.

For other files:
1. After training models in the letter_recog,py, we got Boost.txt and mlp.txt , which are models saved from training Boost and MLP. They will be loaded during handdetect to recognize letters.
2. Handdetect.py is the code that can classify which number you are classifying. (In this case I only consider c and o) 
3. Test.py is the 2 models boost and mlp. When we classify hands in the handdetect.py, it will use the code's of the model in the test.py get the test result and then knowing that which letter it is.
