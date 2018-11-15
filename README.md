# ObjectFingerprint
Identify object surface texture (micrometer level) using computer vision technique.

The input of the algorithm is two object images. The algorithm determines whether the two images are from the same object. 

## Working Scheme

1. Based on the effective circle detection algorithms, all circles in the image are firstly detected. 
Considering the input image contains objects with arbitrary size and angle of view, 
a greedy algorithm should be developed to extract the desired ring of a certain industrial pump, 
which is defined as the ROI for object-fingerprint algorithm.

2. Convolutional neural network is demonstrated powerful for extracting hierarchical
features and encoding multi-level representations. Hence, we mainly focus on utilizing
CNN to learn the unique feature representation for a certain industrial part, rather than
encoding based on hand-crafted and non-robust feature descriptors.

3. Two feature vectors should be extracted simultaneously for two images and then
combined for further comparison. Hence for two major components of the algorithm
(feature extraction module and feature comparison module), we will design different
CNN based structures respectively and then compare their performance.

