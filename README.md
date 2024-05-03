
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Eigenfaces-In-Image-Classification/repo_name">
    <img src="Figures/JHU_LOGO.png" alt="Logo">
  </a>

  <h3 align="center">A Study On The Efficacy of Using Eigenfaces In Image Classification</h3>

  <p align="center">
    The increased use of image classification in technology today has incited a rise in research and development for new approaches in facial detection and identification models. 
    Two common problems in image classification are storing large datasets and model training costs.
    One approach to achieving dimensionality reduction while maintaining performance is Principal Component Analysis where a subset of eigenvectors, also known in the domain of facial detection as ``eigenfaces'', are used to represent the data in a lower dimensionality space.
    This paper presents an image classification model based on eigenfaces and support vector machines using the Amsterdam Dynamic Facial Expression Set (ADFES) dataset. 
    Implementation of an image classification model is described, and performance analysis of the model is presented with a focus on the efficacy of using eigenfaces when training the model.
    <br />
    <a href="https://github.com/Eigenfaces-In-Image-Classification/repo_name"><strong>Explore the Repository »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Eigenfaces-In-Image-Classification/repo_name/issues">Report Bug</a>
  </p>
</p>

Authors: Jacob Taylor Cassady and Dimitri Medina

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#amsterdam-dynamic-facial-expression-set">Amsterdam Dynamic Facial Expression Set</a></li>
    <li><a href="#model">Model</a></li>
    <li><a href="#implementation">Implementation</a></li>
    <li><a href="#analysis">Analysis</a></li>
    <li><a href="#discussion">Discussion</a></li>
    <li><a href="#future-work">Future Work</a></li>
    <li><a href="#acknolwedgements">Acknolwedgements</a></li>
    <li><a href="#referemces">References</a></li>
  </ol>
</details>

## Nomenclature 
| Abbreviation | Definition                                     |
|--------------|------------------------------------------------|
| ADFES        | Amsterdam Dynamic Facial Expression Set        |
| AICE         | Amsterdam Interdisciplinary Centre for Emotion |
| PCA          | Principal Component Analysis                   |
| SVM          | Support Vector Machine                         |

## Introduction
Image classification has become an innovation of interest in recent years due to the parallel advances in machine learning techniques, digital camera technology, and other similar computational fields of science. 
From security monitoring to access control, image classification has become an important tool in many diverse disciplines. 
Two common problems in image classification are storing large datasets and model training costs.

One approach to achieving dimensionality reduction while maintaining a representation of the data is Principal Component Analysis (PCA) [1].
In the field of facial recognition, PCA can remove the reliance on the isolation of features such as eyes and ears, and instead takes a mathematical approach to defining and characterizing face images. 
PCA uses a subset of eigenvectors, also known in facial recognition as ``eigenfaces'', to represent the data in a lower dimensionality space.
If a certain characteristic is prominent in a collection of images, the eigenface corresponding to the characteristic has a larger eigenvalue and represents a higher explained variance of the dataset [2]. 
After utilizing PCA, eigenvectors of lesser importance can be removed to reach a desired level of dimensionality reduction with the trade-off of reduced explained variance of the initial dataset.

SVMs are supervised learning algorithms designed to find the optimal hyperplane that maximizes the margin between the closest data points of opposite classes [3].
SVMs suffer high computational costs when the number of features is large.
Using eigenvectors calculated with PCA has been shown to work well with Support Vector Machines (SVMs) for classification in a variety of domains [4-6].

This paper focuses on analyzing the efficacy of using eigenfaces when performing image classification of the Amsterdam Dynamic Facial Expression Set (ADFES) dataset using SVMs.
Section <a href="#amsterdam-dynamic-facial-expression-set">II</a> provides a description of the ADFES dataset.
Section <a href="#model">III</a> introduces the model used for image classification including the mathematics of eigenfaces and SVMs.
Section <a href="#model">IV</a> describes the implementation of a classification model using eigenfaces and SVMs with the Python programming language.
Section  <a href="#analysis">V</a> presents analysis of the classification model including the efficacy of using eigenfaces in image classification.
This paper concludes with a discussion of the results and future work in sections \ref<a href="#discussion">VI</a> and <a href="#future-work">VII</a> respectively.

## Amsterdam Dynamic Facial Expression Set
<!-- The ADFES dataset was developed by the University of Amsterdam's Amsterdam Interdisciplinary Centre for Emotion (AICE) \cite{adfes}.
The ADFES dataset includes expressions displayed by 22 models from Northern-European and Mediterranean locations.
There are ten emotions included in ADFES dataset: anger, contempt, disgust, embarrassment, fear, joy, neutral, pride, sadness, and surprise.
The ADFES dataset includes both videos and still images.
Each media is labeled with a gender: male or female.
This paper will utilize the 217 still images from the ADFES dataset only.
Figure \ref{fig:ExampleImage} includes an example of a still image from the ADFES dataset.
Table \ref{table:Target Class Distributions} shows the number of classes per target in the dataset and the images per class.
Each image has a width of 720, a height of 576, and three 8-bit color channels: red, green, and blue. -->

## Model

## Implementation

## Analysis

## Discussion

## Future Work

## Appendix

## Acknolwedgements

## References

<ol>
<li>Abdi, H., and Williams, L. J., “Principal component analysis,” Wiley interdisciplinary reviews: computational statistics, Vol. 2, No. 4, 2010, pp. 433–459.</li>
<li>Turk, M. A., and Pentland, A. P., “Face recognition using eigenfaces,” Proceedings. 1991 IEEE computer society conference on computer vision and pattern recognition, IEEE Computer Society, 1991, pp. 586–587.</li>
<li>Hearst, M., Dumais, S., Osuna, E., Platt, J., and Scholkopf, B., “Support vector machines,” IEEE Intelligent Systems and their Applications, Vol. 13, No. 4, 1998, pp. 18–28. https://doi.org/10.1109/5254.708428.</li>
<li>Mangasarian, O. L., and Wild, E. W., “Multisurface proximal support vector machine classification via generalized eigenvalues,” IEEE transactions on pattern analysis and machine intelligence, Vol. 28, No. 1, 2005, pp. 69–74.</li>
<li>Alvarez, I., Górriz, J. M., Ramírez, J., Salas-Gonzalez, D., López, M., Segovia, F., Puntonet, C. G., and Prieto, B., “Alzheimer’s diagnosis using eigenbrains and support vector machines,” Bio-Inspired Systems: Computational and Ambient Intelligence: 10th International Work-Conference on Artificial Neural Networks, IWANN 2009, Salamanca, Spain, June 10-12, 2009. Proceedings, Part I 10, Springer, 2009, pp. 973–980.</li>
<li>Melišek, J. M., and Pavlovicová, M. O., “Support vector machines, PCA and LDA in face recognition,” J. Electr. Eng, Vol. 59, No. 203-209, 2008, p. 1.</li>
<li>van der Schalk, J., Hawk, S. T., Fischer, A. H., and Doosje, B., “Moving faces, looking places: Validation of the Amsterdam Dynamic Facial Expression Set (ADFES),” Emotion, Vol. 11, No. 4, 2011, pp. 907–920. https://doi.org/10.1037/a0023853.</li>
<li>NumFOCUS, Inc., “pandas,” , 2024. URL https://pandas.pydata.org/.</li>
<li>scikit-learn Development Team, “scikit-learn,” , 2024. URL https://scikit-learn.org/stable/.</li>
<li>Halko, N., Martinsson, P.-G., and Tropp, J. A., “Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions,” SIAM review, Vol. 53, No. 2, 2011, pp. 217–288.</li>
<li>Maharana, K., Mondal, S., and Nemade, B., “A review: Data pre-processing and data augmentation techniques,” Global Transitions Proceedings, Vol. 3, No. 1, 2022, pp. 91–99.</li>
<li>Zou, J., and Schiebinger, L., “AI can be sexist and racist—it’s time to make it fair,” , 2018.</li>
<li>NumPy Development Team, “NumPy,” , 2024. URL https://numpy.org/.</li>
<li>Matplotlib Development Team, “matplotlib,” , 2024. URL https://matplotlib.org/.</li>
</ol>
