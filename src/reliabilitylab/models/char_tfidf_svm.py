"""Character n-gram TF-IDF + Linear SVM model."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_char_tfidf_svm() -> Pipeline:
    """Build a character n-gram TF-IDF + Linear SVM pipeline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )