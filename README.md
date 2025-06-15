# faceclust
vgg is th file in which the embeddings where created in 100 sec(colab gpu)
resnet 50 is the the code for quanitation (fp16)

#1000_1000.py
This Python script performs fast and accurate face matching by comparing faces in a large image dataset against one or more reference faces.
It separates all the detected faces and organizes them into individual folders named Person 1, Person 2, Person 3, and so on.

#MTCNN.py
This code is a face matching system that compares faces in a folder of images against a reference face. It detects faces using various detectors (MTCNN, RetinaFace, or YuNet), extracts facial embeddings using FaceNet, and sorts images into "matching" or "non-matching" folders based on cosine similarity with the reference face. The threshold (default 0.7) determines whether a face is considered a match
