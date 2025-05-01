# MachineLearning-Based-Behavioural-Analysis-for-Security-Threat-Detection
With cyber threats increasing at an alarming rate, security breaches have surged by 67% over 
the past five years, and cybercrime is expected to cost the world $10.5 trillion annually by 
2025. Traditional manual threat detection methods rely on rule-based systems and human 
expertise, which are prone to errors, slow in response, and inefficient in handling large-scale 
data. To address these challenges, we propose a machine learning-based behavioral analysis 
approach for security threat detection, leveraging deep learning for improved accuracy. The 
proposed method begins with data preprocessing to clean and normalize the Security Threat 
Dataset, which contains four attack labels: Denial of Service (DoS), Probe, Remote-to-Local 
(R2L), and User-to-Root (U2R). To handle class imbalance, we employ the ADASYN 
(Adaptive Synthetic Sampling) technique, which generates synthetic minority class samples, 
ensuring a balanced dataset. The dataset is then split into training and testing sets to evaluate 
model performance. We compare traditional classifiers such as Naïve Bayes and Support 
Vector Machine (SVM) with our proposed Convolutional Neural Network (CNN)-based 
model, which captures complex patterns and hierarchical representations in network traffic 
data. Experimental results demonstrate that CNN outperforms existing methods in terms of 
detection accuracy, recall, and precision, making it a robust solution for real-time security 
threat analysis.
