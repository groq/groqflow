"""
The following example trains an XGBClassifier against random data
then compares the xgboost result to GroqChip executed via GroqFlow.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier  # pylint: disable=import-error
from groqflow import groqit

batch_size = 320

# Generate random points in a 10-dimensional space with binary labels
np.random.seed(0)
x = np.random.rand(1000, 10).astype(np.float32)
y = np.random.randint(2, size=1000)

# Perform a test/train split of the (random) dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=batch_size, random_state=0
)

# Fit the model using standard sklearn patterns
xgb_model = XGBClassifier(
    n_estimators=10, max_depth=5, random_state=0, objective="binary:logistic"
)
xgb_model.fit(x_train, y_train)

# Build the model
groq_model = groqit(xgb_model, {"input_0": x_test})

# Display a report of standard classifier statistics
print("XGBoost classification report")
print(classification_report(y_test, xgb_model.predict(x_test)))
print("Groq classification report")
print(classification_report(y_test, groq_model.predict(x_test)))
