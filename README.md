## Introduction


This datasets is related to red variants of the Portuguese "Vinho Verde" wine.The dataset describes the amount of various chemicals present in wine and their effect on it's quality. - The datasets can be viewed as classification or regression tasks.
This data frame contains the following columns:
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- quality

In this study, we present a practical application where multiple machine learning models are employed to predict wine taste preferences using readily available analytical data from the certification stage. To determine the optimal parameters with minimal computational effort, we use conventional k-NN algorithm. Knowing that k-NN algorithms under-performs under high dimensional environments, the main purpose of this article is to assess attention model based k-NN by comparing accuracy with other existing models. 

Incorporating the "Attention Is All You Need" mechanism into k-NN can enhance its dimension reduction capabilities by focusing on the most important features in the data. By applying self-attention, the model can learn to weigh the relevance of different features and identify which are most crucial for making predictions. This approach reduces the influence of less important dimensions and helps the k-NN algorithm focus on the most relevant aspects of the dataset. Similar to how attention highlights key relationships in natural language processing, in k-NN, it enables the model to perform more efficiently, improving accuracy and reducing computational complexity in high-dimensional spaces.
