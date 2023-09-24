### Twitter Sentiment Analysis using RNN(Reduced Neural Network) and Machine Learning

#### 1. Introduction

In this project, I used the Twitter Sentiment Analysis Dataset from [Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).

#### 2. Data Preprocessing

```python
eng_stopwords = stopwords.words('english')

def preprocessing_context(context):
    context = re.sub('[^a-zA-Z]', ' ' ,context)
    context = [word.lower() for word in context.split() if word.lower() not in eng_stopwords]
    return ' '.join(context)

data['context'] = data['context'].apply(preprocessing_context)
```

As can be seen from the code above, I removed all the stopwords and punctuations from the context.

#### Model

I used both RNN(Reduced Neural Network) and Machine Learning to train the model.

#### Results

- Machine Learning

  | Model               | Accuracy |
  | ------------------- | -------- |
  | Logistic Regression | 0.83     |
  | SVC                 | 0.86     |

- RNN(Reduced Neural Network)
  | Activation | Accuracy |
  | ----- | -------- |
  | Tanh | 0.87 |
  | ReLU | 0.85 |
  | Sigmoid | 0.29 |
