# Determining-Human-Intelligence-Using-Brain-Network-Analysis
In this project, our task is to predict whether a person has high or low math capability, using his/her FSIQ (Full Scale Intelligence Quotient) score. 

How to run the project:
1. pip install -r requirements.txt
2. python main.py 
Note: Place all the code files in a single document

Project Details:

The main goal of this project is to predict a person’s math capability by analyzing their brain network. We have considered human subjects with an FSIQ score greater than 120 as highly math capable while the rest of the subjects are classified as having normal math capability. Supervised learning algorithms were implemented to perform this binary classification. Topological features of the brain network graph were extracted. Using this, we found the most pivotal nodes/edges of the graph with respect to our classification problem. The feature values at the pivotal nodes were used to classify a brain network corresponding to a human subject as having highly math capability or normal math capability.

Therefore, the project is composed of three main tasks:
1) Extraction of topological features from the brain network graph.
2) Determining the pivotal nodes/edges for our classification problem from the graph.
3) Classification of the human subjects as highly math capable or not using feature values of the pre-determined pivotal nodes/edges.

Classifiers Used: (We have used 3 classifiers in this project)
1. SVM (Support Vector Machine)
2. Logistic Regression
3. kNN (k-Nearest Neighbours)

Results: 

We tried different models like SVM, Logistic Regression, KNN to predict the math capability of the human subjects. We found that SVM provides the highest prediction accuracy among the other classifiers. But as the dataset is not big enough, the results are not satisfying enough to apply it in real world.

It is important to use larger dataset for training and testing the classifier as the dataset we have used is small and may not have enough discriminating power to classify the features. It would also be useful to use advanced learning algorithms like ensemble learning and deep learning methods for classification. We could extend our analysis to study the differences in creativity, neuroticism, agreeability etc.	

References:

[1]Vivek Kulkarni, Jagat Sastry, “Sex differences in the Human Connectome”, International   Conference on Brain and Health Informatics BHI 2013: Brain and Health Informatics pp 82-91

[2] Shai Shalev-Shwartz, Yoram Singer, Nathan Srebro, Andrew Cotter, “Pegasos: Primal Estimated sub-GrAdient SOlver for SVM”, Mathematical Programming, March 2011, Volume 127, Issue 1, pp 3–30.


