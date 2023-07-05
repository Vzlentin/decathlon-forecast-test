# decathlon-forecast-test

[Instructions](https://github.com/Vzlentin/decathlon-forecast-test/blob/main/MLE%20Forecast%20-%20Technical%20test.pdf)

Answers to part 1 and 2 are [here](https://github.com/Vzlentin/decathlon-forecast-test/blob/main/notebooks/answers.ipynb)

Part 3:

a-d The pipeline is [here](https://github.com/Vzlentin/decathlon-forecast-test/blob/main/pipeline.py)

e. Deploying machine learning models involves several critical steps and potential challenges:

Data Quality: Ensuring good data quality is fundamental, as models' performance heavily relies on the data they're trained on.

Overfitting: Models can overfit to training data, reducing their ability to generalize to new data. Techniques to mitigate overfitting are essential.

Model Interpretability: High-performing models, especially deep learning ones, can lack interpretability, which might be problematic in domains requiring explainability.

Computational Constraints: The heavy computational requirements for training and deploying models can pose challenges in resource-limited or real-time scenarios.

Model Drift: Models can degrade over time if data distributions change, requiring ongoing monitoring and updating.

Scalability: Models need to perform well not just on small datasets but also when scaled to larger datasets or real-time environments.

Maintenance: Models require continuous monitoring, updating, and occasional retraining, making maintenance a crucial part of the deployment process.

f. To monitor the pipeline in production, we could report the validation rmse at each re-training and create an alert based on a threshold.
This is easy to implement in Data Science platforms such as Dataiku.
