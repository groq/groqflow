"""
The following example trains a RandomForestClassifier against random data
then compares the sklearn result to GroqChip executed via GroqFlow.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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
skl_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
skl_model.fit(x_train, y_train)

# Build the model
groq_model = groqit(skl_model, {"input_0": x_test})

# Display a report of standard classifier statistics
print("SKLearn classification report")
print(classification_report(y_test, skl_model.predict(x_test)))
print("Groq classification report")
print(classification_report(y_test, groq_model.predict(x_test)))

print("Example randomforest.py finished")
