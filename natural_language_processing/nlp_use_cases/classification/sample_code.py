from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Assuming you have your text data (X) and labels (y)
# X would be your email messages, y would be 'spam' or 'ham'

# 1. Convert text to numerical features (e.g., TF-IDF)
# This is crucial before applying SMOTE
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X_text_data)

# 2. Split data into training and testing sets *before* applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

# Check original class distribution in training set
print("Original training set distribution:", Counter(y_train))

# 3. Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution
print("Resampled training set distribution:", Counter(y_train_resampled))

# 4. Train your Decision Tree model on the resampled data
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_resampled, y_train_resampled)

# 5. Evaluate on the *original* (unresampled) test set
y_pred = dt_classifier.predict(X_test)
print("\nClassification Report after SMOTE:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
