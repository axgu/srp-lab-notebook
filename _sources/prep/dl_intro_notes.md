# Intro
### Basics of Convolutional Neural Networks (CNN)
* Artificial neural networks made of layers
    * Performs single image processing function
    * Convolution: summarization of each region of the image or matrix
        * Flashlight metaphor: 
            * Filter/neuron/kernel - array of numbers representing weights
            * Sliding motion - convolution
            * Illuminated region - receptive field
        * Multiply values in the filter with original pixel values; sum result
        * Feature map of smaller size is created
        * Stack different convolution modules
            * Units cover larger zones of input from early to late layers - tolerance to spacial translation of features/shape in image, so later layers can identify patterns or shapes independently of the original location of the shape
            * Features are more complex from early to late layers
    * Feature detectors that receive input and pass information to the next
* Resemblance to primate visualization system

### CNN Learning and Training
* Network receives input and produces an output related to the network’s task (image classification, etc.)
* Weights are initially set randomly
* Supervised network - give network the correct answer
    * Working network gives a probability between 0 and 1 for each label
    * Cost = 𝛴(Network’s answer - wanted answer)^2
    * Backpropagation: weights changed by sending cost back through the network
        * Network iteratively calculates error values for each layer; updates parameters
    * Show images of all categories to teach network
    * Each layer learns more complex patterns: contours => shapes => objects

### Visualizations of CNN units
* Analyze what the network has learned
* Inspiration from neuroscience: show specific image/pattern to a brain while recording a response of the cell
    * Black-box
* For artificial neural network, show a lot of different patterns and record responses of units of neural networks to try to determine sensitivities of each cell
* Estimating receptive fields:
    * Identify regions of the image that lead to high unit activations
    * Sliding-window stimuli contains small randomized patch at different spatial locations
    * Feed occluded images into same network
    * Record change in activation as compared to original image
        * Large discrepancy = given patch is important
    * Obtain discrepancy map for each unit of each image shown
    * Re-center discrepancy map and average calibrated discrepancy maps to generalize final receptive field for that unit
* Train same network to solve different tasks
* Find discriminative features relevant to categorization tasks
* Network plasticity/fine-tuning - what happens to neurons when they relearn
    * Features forgotten

### Comparison Natural (Brain) vs Artificial Neural Networks
* Use artificial networks to learn about how biological neural networks work
* Correspondence between response to visual objects
* Algorithmic-specific fMRI searchlight analysis
    * Move spherical of cortex patchy searchlight through brain volume to select location of local set of voxels
        * Vector corresponds to activity of patch
        * Build matrix of dissimilarity between pairs of image response
    * Present object images shown to humans in fMRI to neural network
    * Extract responses over all units of layers of all images to build matrices of similarity for each layer
    * Compare by taking Spearman correlation between matrices
    * Representation Similarity Analysis (RSA): compare responses from different sensors/data sources by measuring difference between responses
* Spatiotemporal maps of correlations between human brain and model layers

## Resources
* [Neuromatch Academy: Computational Neuroscience](https://compneuro.neuromatch.io/tutorials/W1D5_DeepLearning/student/W1D5_Intro.html)