"""TF-IDF + Linear SVM baseline for intent classification."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_tfidf_svm() -> Pipeline:
    """Build a TF-IDF + Linear SVM classification pipeline."""

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    max_iter=5000,
                ),
            ),
        ]
    )

    return model