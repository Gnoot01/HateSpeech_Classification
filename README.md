The challenge was to classify hate speech data **scored by macro F1** and constrained to using **only non-deep learning methods** with preprocessed tf-idf tokens, as part of my Machine Learning course, on Kaggle.

The rationale to constrain to only non-deep learning methods was to explore and understand the fundamentals of traditional machine learning models and preprocessing techniques, rather than relying solely on increased GPU compute power using state-of-the-art.

Our final model was a soft voting ensemble that aggregates the prediction probabilities of several base models with fined-tuned hyperparameters. We selected a diverse set of base models including Logistic Regression, Naive Bayes, and boosting models. 

- Emerged the winner out of 36 teams in the cohort of 160 students and 1 of the only 2 teams to beat the Blue Line, the upper benchmark set by instructors
- Gained hands-on experience in investigating and addressing data challenges, including sparsity, distribution, class imbalance, and correlations
- Explored applying advanced data augmentation techniques using generative Gaussian Mixture Models, feature transformation techniques using Yeo-Johnson Power Transformer and Quantile Transformer, dimensionality reduction using PCA and SVD, model hyperparameter optimization using Bayesian Optimization and RandomizedSearchCV
- Explored advanced ensemble learning techniques such as stacking, blending, and cascading to enhance model robustness
- Sourced and configured GPU-accelerated alternatives like cuML, to speed up computation

Feel free to check out the implementations and the thought processes behind them! 💡
