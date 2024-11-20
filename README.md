# **Enhancing MBTI Prediction and Sentiment Analysis with Advanced Language Models**
This repository contains the implementation of a project that applies LLMs(BERT, RoBERTa, DistilBERT,XLNET,GPT-2) to predict Myers-Briggs Type Indicator (MBTI) personality types from textual data, specifically forum posts. The project explores state-of-the-art NLP techniques to classify individuals into one of 16 MBTI types based on their text content.
## **Project Overview**
The Myers-Briggs Type Indicator (MBTI) categorizes individuals into 16 personality types based on preferences in four dimensions:

- Extraversion (E) vs. Introversion (I)
- Sensing (S) vs. Intuition (N)
- Thinking (T) vs. Feeling (F)
- Judging (J) vs. Perceiving (P)

For example, someone who is introverted, sensing, feeling, and perceiving would be classified as ISFP.

This project leverages transformer-based models, including BERT, RoBERTa, DistilBERT, XLNet, and GPT-2, to predict MBTI personality types from textual data. By fine-tuning these models on an MBTI dataset, we built accurate classifiers capable of analyzing forum posts and predicting the corresponding personality type.

## **Key Features**
- Advanced NLP Models: Utilized state-of-the-art transformers like BERT, RoBERTa, XLNet, and GPT-2.
- Hyperparameter Tuning: Conducted iterative experiments to optimize model performance.
- Evaluation Metrics: Assessed models using accuracy, F1-score, confusion matrices, and user input testing.
- Performance Insights:
  - XLNet achieved 68% accuracy, the highest among all models.
  - Other models, including GPT-2, achieved accuracy around 66%.
  - Deeper Analysis: Confusion matrices were used to analyze performance for each personality type and their corresponding four labels.
    
## **Applications**
This project offers valuable applications in:
- Personalized Marketing: Understanding customer personality traits for targeted marketing.
- Social Media Analysis: Profiling users based on personality for sentiment and trend analysis.
- Psychology: Assisting in personality research and behavioral studies.

## **Technologies Used**
Programming Language: Python
Libraries:
- Hugging Face Transformers
- PyTorch
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## **Dataset**
The project used the [MBTI Kaggle Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type/data) , containing forum posts labeled with MBTI personality types. The dataset was preprocessed and prepared it for training.

## **Results**
This section presents the outcomes of training and evaluating the transformer-based models on the MBTI dataset. We assessed the models' performance using metrics such as accuracy, classification reports (including precision, recall, and F1-score), and confusion matrices for personality types and their corresponding four labels. Additionally, the models were tested with user-provided input to determine their ability to predict personality types effectively.

Model Performance Overview
- XLNet emerged as the most consistent performer, achieving a balanced accuracy across most personality types. While GPT-2 slightly outperformed XLNet in overall accuracy 
  (68% compared to XLNet’s 67%), XLNet demonstrated better performance consistency across various personality types.
- Other models, including BERT, RoBERTa, and DistilBERT, achieved accuracies of 66.46%, 65.99%, and 66.05%, respectively.

Detailed Personality Type Analysis

Each model showed varying strengths for different personality types:

- XLNet excelled in predicting INTJ, INFP, and ISFP types, with the highest accuracy of 82.97% for INFP.
- GPT-2 performed exceptionally well for INFJ, INTP, and ISTP types.
- BERT and RoBERTa achieved the highest accuracy for ENTP and ENFP, while DistilBERT outperformed others for ENFJ and ESTP.
- The least represented type in the dataset, ESFP, was best predicted by GPT-2 and XLNet, although accuracy remained low due to the dataset's imbalance.

Insights from Confusion Matrices
- For INFP, the most frequent personality type (21% of the dataset), XLNet demonstrated the highest true positives, correctly identifying 307 instances.
- GPT-2 and XLNet achieved the best results for the ESFP type, despite its lower representation.
- Most misclassifications occurred in the Judging/Perceiving (J/P) labels, with 192 instances of J predicted as P and 142 instances of P predicted as J. This indicates 
  challenges in distinguishing between these traits.

### **User Input Evaluation**
To further validate model performance, the trained models were tested with user-generated input resembling tweets from an individual with INFP traits. All models except DistilBERT correctly predicted the personality type as INFP. DistilBERT misclassified the input as INFJ, likely due to its lower accuracy for INFP predictions and misclassifications observed in the testing phase (41 instances of INFP misclassified as INFJ).
