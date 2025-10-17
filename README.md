# Fake News Detection Using Machine Learning

A comprehensive machine learning pipeline for automatically detecting fake news articles using Natural Language Processing (NLP) and multiple classification algorithms.

## Author

**Adam Lubinsky**  
Email: alubinsky1728@gmail.com  
LinkedIn: [linkedin.com/in/adam-lubinsky-32b2b9337](https://www.linkedin.com/in/adam-lubinsky-32b2b9337/)  
GitHub: [@Alubinsky](https://github.com/Alubinsky)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Methodology](#methodology)
- [Real-World Applications](#real-world-applications)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This project implements a sophisticated fake news detection system that achieves over 90% accuracy in distinguishing between fake and real news articles. With the proliferation of misinformation across digital platforms, automated detection systems are crucial for maintaining information integrity.

The system analyzes linguistic patterns, writing styles, and content features to classify news articles as either FAKE or REAL, providing insights into the characteristics that distinguish legitimate journalism from misinformation.

## Features

- **Multiple ML Models**: Compares Naive Bayes, Logistic Regression, and Random Forest classifiers
- **Advanced Text Processing**: TF-IDF vectorization with bigram support
- **Hyperparameter Optimization**: Grid search for optimal model parameters
- **Comprehensive Analysis**: 
  - Confusion matrices
  - Feature importance visualization
  - Classification reports with precision, recall, and F1-scores
- **Comparative Experiments**:
  - Impact of stopword removal
  - Title vs. full text vs. combined analysis
- **Professional Visualizations**: Publication-ready charts and graphs
- **Detailed Metrics**: 95% confidence intervals for accuracy measurements

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository
```bash
git clone https://github.com/Alubinsky/Fake-Or-Real-News.git
cd Fake-Or-Real-News
```

2. Install required packages
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
```

## Usage

### Quick Start

Run the complete analysis pipeline:
```bash
python real_or_fake_news.py
```

### Google Colab

You can also run this project directly in Google Colab:

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `real_or_fake_news.py` or copy the code
3. Run all cells

### Output

The script generates:

- **Console output**: Detailed metrics and progress updates
- **Visualizations**:
  - `confusion_matrix.png` - Model prediction accuracy breakdown
  - `feature_importance.png` - Most indicative words for each class
  - `text_source_comparison.png` - Performance comparison across text sources
- **CSV file**: `model_performance_summary.csv` - Comprehensive results summary

## Project Structure
```
Fake-Or-Real-News/
│
├── real_or_fake_news.py          # Main analysis script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── outputs/                       # Generated outputs (created on first run)
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── text_source_comparison.png
│   └── model_performance_summary.csv
│
└── notebooks/                     # Jupyter notebooks (optional)
    └── fake_news_analysis.ipynb
```

## Results

### Model Performance

| Model | Accuracy | 95% Confidence Interval |
|-------|----------|-------------------------|
| Naive Bayes (Tuned) | 92.3% | (90.1%, 94.2%) |
| Random Forest | 91.8% | (89.5%, 93.8%) |
| Logistic Regression | 91.2% | (88.9%, 93.2%) |
| Naive Bayes (Base) | 90.7% | (88.3%, 92.8%) |

*Results may vary slightly based on random seed and dataset sampling*

### Key Findings

1. **Text vs. Title Performance**:
   - Full text: approximately 92% accuracy
   - Title only: approximately 78% accuracy
   - Combined: approximately 93% accuracy

2. **Linguistic Patterns**:
   - Fake news uses more sensational language
   - Real news has more formal structure and specific attributions
   - Emotional appeals are stronger indicators of misinformation

3. **Feature Importance**:
   - Bigrams (word pairs) improve accuracy by approximately 3-5%
   - Stopword removal has minimal impact (approximately 0.5% difference)

## Methodology

### 1. Data Preprocessing

- Text normalization (lowercase, special character removal)
- Label encoding (FAKE to 0, REAL to 1)
- Train/test split (80/20) with stratification
- Sample size: 3,000 articles

### 2. Feature Extraction

**TF-IDF Vectorization** with:
- Unigrams and bigrams (1-2 word phrases)
- 5,000 maximum features
- English stopword removal
- Min/max document frequency filtering

### 3. Model Training

- Multiple algorithm comparison
- 5-fold cross-validation
- Hyperparameter tuning via grid search
- Stratified sampling to maintain class balance

### 4. Evaluation

- Accuracy with 95% confidence intervals (Wilson score method)
- Precision, recall, F1-score
- Confusion matrix analysis
- Feature importance examination

### TF-IDF Explained

TF-IDF stands for Term Frequency-Inverse Document Frequency. It measures how important a word is to a document:

- **Term Frequency (TF)**: How often a word appears in a document
- **Inverse Document Frequency (IDF)**: How rare/common a word is across all documents

**Formula**: TF-IDF = (word frequency in document) × log(total documents / documents containing word)

Words that appear frequently in a document but rarely across all documents receive high TF-IDF scores, making them highly informative for classification.

## Real-World Applications

This fake news detection system can be applied to:

### Social Media Platforms
- Automated content moderation
- Flagging suspicious content for human review
- Reducing spread of misinformation

### News Aggregators
- Source credibility ranking
- Content quality filtering
- Trusted source prioritization

### Browser Extensions
- Real-time fact-checking tools
- Warning labels for questionable content
- User empowerment for informed decisions

### Research
- Studying misinformation spread patterns
- Analyzing linguistic characteristics of fake news
- Understanding psychological factors in belief formation

### Education
- Media literacy training tools
- Teaching critical thinking skills
- Demonstrating NLP and machine learning concepts

## Future Improvements

### Technical Enhancements

- Implement deep learning models (BERT, RoBERTa, GPT-based classifiers)
- Add multi-language support
- Incorporate image and video analysis
- Develop real-time classification API
- Create interactive web dashboard
- Expand dataset size for improved generalization

### Feature Expansions

- Source credibility scoring and publisher reputation tracking
- Temporal trend analysis and tracking emerging narratives
- Network propagation patterns and viral spread analysis
- Claim extraction and automated fact verification
- Author profiling and writing style analysis
- Cross-reference verification across multiple sources

### Model Improvements

- Ensemble methods combining multiple models
- Active learning for continuous improvement
- Domain adaptation for different news categories
- Adversarial training to handle sophisticated fake news
- Explainable AI for transparency in predictions

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- Bug fixes and code optimization
- Additional ML models and algorithms
- Improved visualizations
- Documentation improvements
- Test coverage expansion
- New feature implementations
- Dataset expansion and curation

### Code Style

- Follow PEP 8 style guidelines
- Include docstrings for functions and classes
- Add comments for complex logic
- Write unit tests for new features

## Limitations and Ethical Considerations

### Limitations

1. **Context Dependency**: Model detects linguistic patterns, not factual accuracy
2. **Adversarial Vulnerability**: Sophisticated actors can adapt to detection systems
3. **Dataset Bias**: Performance depends on training data quality and diversity
4. **Language Specificity**: Currently trained on English news articles only
5. **Temporal Drift**: Misinformation tactics evolve, requiring regular model updates

### Ethical Considerations

- **False Positives**: Risk of labeling legitimate news as fake
- **Censorship Concerns**: Automated systems require transparency and accountability
- **Political Neutrality**: Regular auditing needed to prevent partisan bias
- **Privacy**: Text analysis must respect user data rights
- **Human Oversight**: AI should assist, not replace, human judgment

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: [Fake News Dataset](https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv)
- **Libraries**: scikit-learn, pandas, matplotlib, seaborn, statsmodels
- **Community**: Open-source contributors and researchers in NLP and misinformation detection

## References

### Academic Papers

- Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2018). "Automatic Detection of Fake News"
- Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). "Fake News Detection on Social Media: A Data Mining Perspective"
- Zhou, X., & Zafarani, R. (2020). "A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities"

### Tools and Resources

- [scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [Natural Language Toolkit (NLTK)](https://www.nltk.org/)

## Contact

Have questions or suggestions? Reach out!

- **Email**: alubinsky1728@gmail.com
- **LinkedIn**: [Adam Lubinsky](https://www.linkedin.com/in/adam-lubinsky-32b2b9337/)
- **GitHub Issues**: [Report a bug or request a feature](https://github.com/Alubinsky/Fake-Or-Real-News/issues)

---

**Made by Adam Lubinsky**
