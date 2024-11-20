## **Enhancing MBTI Prediction and Sentiment Analysis with Advanced Language Models**
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
