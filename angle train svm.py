import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Define your data
loaded_array = np.load(r'C:\Users\psaar\Documents\biceps\angles.npy') 
arrays=loaded_array[:88]
print(arrays.shape)  # Print the shape of the loaded array 
labels=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,])
print(len(labels))

# Split data into training and testing sets
train_arrays, test_arrays, train_labels, test_labels = train_test_split(arrays, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
train_arrays_scaled = scaler.fit_transform(train_arrays)
test_arrays_scaled = scaler.transform(test_arrays)

# Create an SVM classifier
svm = SVC(kernel='linear')

# Train the SVM classifier
svm.fit(train_arrays_scaled, train_labels)

# Predict on the test set
predictions = svm.predict(test_arrays_scaled)

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
print(f'Test accuracy: {accuracy:.4f}')
joblib.dump(svm, 'model2.pkl')
