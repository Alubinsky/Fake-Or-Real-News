# -*- coding: utf-8 -*-
"""
Real Or Fake News Detection - Enhanced Version
Author: Adam Lubinsky
Email: alubinsky1728@gmail.com
LinkedIn: https://www.linkedin.com/in/adam-lubinsky-32b2b9337/
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from statsmodels.stats.proportion import proportion_confint
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FAKE NEWS DETECTION SYSTEM")
print("="*60)
print("✓ All libraries imported successfully\n")

# ============================================================================
# 2. LOAD AND PREPROCESS DATA
# ============================================================================

print("="*60)
print("STEP 1: DATA LOADING & PREPROCESSING")
print("="*60 + "\n")

# Load dataset and shuffle
df = pd.read_csv('https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv')
df = shuffle(df, random_state=42)
print(f"✓ Data loaded: {len(df)} total samples")

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop rows with missing 'text' or 'label'
df = df.dropna(subset=['text', 'label'])
print(f"✓ After dropping nulls: {len(df)} samples")

# Use MORE data for better performance (increase from 1000 to 3000)
sample_size = min(3000, len(df))  # Use 3000 samples or max available
df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
print(f"✓ Using {sample_size} samples for training")

# Text preprocessing function
def preprocess_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()  # Lowercase
    text = text.replace('[^\w\s]', ' ')  # Remove special characters
    return text

# Apply preprocessing
print("\nPreprocessing text...")
df['text_clean'] = df['text'].apply(preprocess_text)
df['title_clean'] = df['title'].apply(preprocess_text)

# Encode labels: FAKE -> 0, REAL -> 1
df['label_num'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# Show label distribution
print("\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nBalance: {df['label'].value_counts(normalize=True).round(3).to_dict()}")

# ============================================================================
# 3. TRAIN/TEST SPLIT
# ============================================================================

print("\n" + "="*60)
print("STEP 2: TRAIN/TEST SPLIT")
print("="*60 + "\n")

X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

print(f"✓ Train samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")
print(f"\nTrain label distribution:")
print(y_train.value_counts())

# ============================================================================
# 4. TEXT VECTORIZATION WITH TF-IDF
# ============================================================================

print("\n" + "="*60)
print("STEP 3: TEXT VECTORIZATION (TF-IDF)")
print("="*60 + "\n")

# Create vectorizer with optimized parameters
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2),  # Include bigrams for better context
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.8  # Ignore terms that appear in more than 80% of documents
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✓ Vectorized feature dimensions: {X_train_vec.shape[1]}")
print(f"✓ Sparsity: {(1 - X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1])):.2%}")
print(f"\nFirst 10 features: {list(vectorizer.get_feature_names_out()[:10])}")

# ============================================================================
# 5. TRAIN MULTIPLE CLASSIFIERS
# ============================================================================

print("\n" + "="*60)
print("STEP 4: TRAINING MULTIPLE CLASSIFIERS")
print("="*60 + "\n")

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}

print("Training models...\n")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate confidence interval
    n = len(y_test)
    lower, upper = proportion_confint(count=int(accuracy*n), nobs=n, alpha=0.05, method='wilson')
    
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'accuracy': accuracy,
        'ci_lower': lower,
        'ci_upper': upper
    }
    
    print(f"  ✓ Accuracy: {accuracy:.4f} (95% CI: {lower:.4f} - {upper:.4f})")

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
for name, res in results.items():
    print(f"{name:25s}: {res['accuracy']:.4f}")

# ============================================================================
# 6. HYPERPARAMETER TUNING FOR NAIVE BAYES
# ============================================================================

print("\n" + "="*60)
print("STEP 5: HYPERPARAMETER TUNING")
print("="*60 + "\n")

print("Performing hyperparameter tuning for Naive Bayes...")

param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]  # Smoothing parameter
}

grid_search = GridSearchCV(
    MultinomialNB(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_vec, y_train)

print(f"✓ Best alpha: {grid_search.best_params_['alpha']}")
print(f"✓ Best CV accuracy: {grid_search.best_score_:.4f}")

# Use the best model
best_nb = grid_search.best_estimator_
y_pred_tuned = best_nb.predict(X_test_vec)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"✓ Tuned Naive Bayes test accuracy: {accuracy_tuned:.4f}")

# Update results
results['Naive Bayes (Tuned)'] = {
    'model': best_nb,
    'predictions': y_pred_tuned,
    'accuracy': accuracy_tuned
}

# ============================================================================
# 7. DETAILED PERFORMANCE ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("STEP 6: DETAILED PERFORMANCE ANALYSIS")
print("="*60 + "\n")

# Choose the best performing model
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model_results = results[best_model_name]

print(f"BEST MODEL: {best_model_name}")
print(f"Accuracy: {best_model_results['accuracy']:.4f}\n")

print("Classification Report:")
print(classification_report(
    y_test, 
    best_model_results['predictions'], 
    target_names=['FAKE', 'REAL'],
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(y_test, best_model_results['predictions'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['FAKE', 'REAL'], 
            yticklabels=['FAKE', 'REAL'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
print(f"\nDetailed Metrics:")
print(f"  True Negatives (Correct FAKE):  {tn}")
print(f"  False Positives (FAKE as REAL): {fp}")
print(f"  False Negatives (REAL as FAKE): {fn}")
print(f"  True Positives (Correct REAL):  {tp}")
print(f"\n  Precision (REAL): {tp/(tp+fp):.4f}")
print(f"  Recall (REAL):    {tp/(tp+fn):.4f}")
print(f"  F1-Score (REAL):  {2*tp/(2*tp+fp+fn):.4f}")

# ============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
print("="*60 + "\n")

# Get the best Naive Bayes model (works best for feature analysis)
nb_model = results['Naive Bayes (Tuned)']['model'] if 'Naive Bayes (Tuned)' in results else results['Naive Bayes']['model']

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Get log probabilities for each class
fake_log_probs = nb_model.feature_log_prob_[0]
real_log_probs = nb_model.feature_log_prob_[1]

# Top 15 words for each class
n_top = 15

print("TOP WORDS INDICATING FAKE NEWS:")
print("-" * 50)
fake_indices = np.argsort(fake_log_probs)[-n_top:][::-1]
for i, idx in enumerate(fake_indices, 1):
    print(f"{i:2d}. {feature_names[idx]:20s} (score: {fake_log_probs[idx]:.4f})")

print("\n" + "TOP WORDS INDICATING REAL NEWS:")
print("-" * 50)
real_indices = np.argsort(real_log_probs)[-n_top:][::-1]
for i, idx in enumerate(real_indices, 1):
    print(f"{i:2d}. {feature_names[idx]:20s} (score: {real_log_probs[idx]:.4f})")

# Visualize top words
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# FAKE news words
fake_words = [feature_names[i] for i in fake_indices]
fake_scores = [fake_log_probs[i] for i in fake_indices]
ax1.barh(range(n_top), fake_scores, color='#ff6b6b')
ax1.set_yticks(range(n_top))
ax1.set_yticklabels(fake_words)
ax1.set_xlabel('Log Probability')
ax1.set_title('Top Words in FAKE News', fontweight='bold')
ax1.invert_yaxis()

# REAL news words
real_words = [feature_names[i] for i in real_indices]
real_scores = [real_log_probs[i] for i in real_indices]
ax2.barh(range(n_top), real_scores, color='#51cf66')
ax2.set_yticks(range(n_top))
ax2.set_yticklabels(real_words)
ax2.set_xlabel('Log Probability')
ax2.set_title('Top Words in REAL News', fontweight='bold')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature importance chart saved as 'feature_importance.png'")
plt.show()

# ============================================================================
# 9. EXPERIMENT: STOPWORD REMOVAL COMPARISON
# ============================================================================

print("\n" + "="*60)
print("STEP 8: EXPERIMENT - STOPWORD REMOVAL")
print("="*60 + "\n")

# Without stopwords (already done above)
accuracy_with = results['Naive Bayes']['accuracy']

# WITHOUT removing stopwords
vectorizer_no_stop = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_no_stop = vectorizer_no_stop.fit_transform(X_train)
X_test_no_stop = vectorizer_no_stop.transform(X_test)

clf_no_stop = MultinomialNB()
clf_no_stop.fit(X_train_no_stop, y_train)
y_pred_no_stop = clf_no_stop.predict(X_test_no_stop)
accuracy_without = accuracy_score(y_test, y_pred_no_stop)

print(f"Accuracy WITH stopword removal:    {accuracy_with:.4f}")
print(f"Accuracy WITHOUT stopword removal: {accuracy_without:.4f}")
print(f"Difference: {abs(accuracy_with - accuracy_without):.4f}")

if accuracy_with > accuracy_without:
    print("✓ Stopword removal IMPROVED performance")
elif accuracy_without > accuracy_with:
    print("✗ Stopword removal DECREASED performance")
else:
    print("→ No significant difference")

# ============================================================================
# 10. EXPERIMENT: TITLE VS FULL TEXT VS COMBINED
# ============================================================================

print("\n" + "="*60)
print("STEP 9: EXPERIMENT - TITLE VS TEXT")
print("="*60 + "\n")

# Title-only classification
X_train_title, X_test_title, y_train_title, y_test_title = train_test_split(
    df['title_clean'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

vectorizer_title = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_train_title_vec = vectorizer_title.fit_transform(X_train_title)
X_test_title_vec = vectorizer_title.transform(X_test_title)

clf_title = MultinomialNB()
clf_title.fit(X_train_title_vec, y_train_title)
y_pred_title = clf_title.predict(X_test_title_vec)
accuracy_title = accuracy_score(y_test_title, y_pred_title)

# Combined title + text
df['combined'] = df['title_clean'] + ' ' + df['text_clean']
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
    df['combined'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

vectorizer_combined = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_train_combined_vec = vectorizer_combined.fit_transform(X_train_combined)
X_test_combined_vec = vectorizer_combined.transform(X_test_combined)

clf_combined = MultinomialNB()
clf_combined.fit(X_train_combined_vec, y_train_combined)
y_pred_combined = clf_combined.predict(X_test_combined_vec)
accuracy_combined = accuracy_score(y_test_combined, y_pred_combined)

# Compare results
accuracy_text = results['Naive Bayes']['accuracy']

print(f"Accuracy using TITLE only:      {accuracy_title:.4f}")
print(f"Accuracy using TEXT only:       {accuracy_text:.4f}")
print(f"Accuracy using TITLE + TEXT:    {accuracy_combined:.4f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
methods = ['Title Only', 'Text Only', 'Title + Text']
accuracies = [accuracy_title, accuracy_text, accuracy_combined]
colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']

bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Comparison: Different Text Sources', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontweight='bold')

plt.legend()
plt.tight_layout()
plt.savefig('text_source_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Comparison chart saved as 'text_source_comparison.png'")
plt.show()

# ============================================================================
# 11. FINAL SUMMARY AND VALIDATION
# ============================================================================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60 + "\n")

print("Model Performance Rankings:")
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for i, (name, res) in enumerate(sorted_results, 1):
    print(f"{i}. {name:25s}: {res['accuracy']:.4f}")

print(f"\n✓ Best Model: {sorted_results[0][0]} with {sorted_results[0][1]['accuracy']:.4f} accuracy")
print(f"✓ Dataset Size: {len(df)} samples")
print(f"✓ Train/Test Split: {len(X_train)}/{len(X_test)}")
print(f"✓ Feature Dimensions: {X_train_vec.shape[1]}")

# Validation Tests
print("\n" + "="*60)
print("VALIDATION TESTS")
print("="*60)

try:
    assert set(df['label_num'].unique()) == {0, 1}, "Labels must be 0 or 1"
    print("✓ Label encoding validated")
    
    assert len(X_train) + len(X_test) == len(df), "Train/test split sum error"
    print("✓ Train/test split validated")
    
    for name, res in results.items():
        assert set(res['predictions']).issubset({0, 1}), f"{name} predictions invalid"
    print("✓ All predictions validated")
    
    assert all(res['accuracy'] >= 0.5 for res in results.values()), "Model performing worse than random"
    print("✓ All models beat random baseline")
    
    print("\n" + "="*60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("="*60)
    
except AssertionError as e:
    print(f"\n✗ TEST FAILED: {e}")

# ============================================================================
# 12. SAVE RESULTS
# ============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60 + "\n")

# Save model performance summary
summary_df = pd.DataFrame([
    {'Model': name, 'Accuracy': res['accuracy']} 
    for name, res in results.items()
]).sort_values('Accuracy', ascending=False)

summary_df.to_csv('model_performance_summary.csv', index=False)
print("✓ Model summary saved as 'model_performance_summary.csv'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  1. confusion_matrix.png")
print("  2. feature_importance.png")
print("  3. text_source_comparison.png")
print("  4. model_performance_summary.csv")
print("\nAuthor: Adam Lubinsky")
print("Email: alubinsky1728@gmail.com")
print("LinkedIn: https://www.linkedin.com/in/adam-lubinsky-32b2b9337/")
print("="*60)
