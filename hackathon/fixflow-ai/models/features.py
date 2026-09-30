"""FixFlow AI Feature Engineering Pipeline.

Extracts TF-IDF text features and structured numerical/categorical features
from complaint data, combining them into a single feature matrix.
"""

import numpy as np
import pandas as pd
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# Fixed category lists for one-hot encoding
LOCATION_ZONES = ['Academic Zone', 'Administrative Zone', 'Hostel Zone', 'Public Zone', 'Residential Zone']
TIME_OF_DAY = ['afternoon', 'evening', 'morning', 'night']
WEATHER_CONDITIONS = ['Clear', 'Cloudy', 'Heavy Rain', 'Light Rain', 'Storm']
CATEGORIES = [
    'Building Damage', 'Drainage Blockage', 'Electrical Hazard', 'Public Safety',
    'Road Damage', 'Streetlight Failure', 'Waste Management', 'Water Leakage'
]


def build_tfidf_vectorizer(
    texts: list[str],
    max_features: int = 300,
    ngram_range: tuple[int, int] = (1, 2)
) -> TfidfVectorizer:
    """Build and fit a TF-IDF vectorizer on the given texts.

    Args:
        texts: List of complaint text strings to fit on.
        max_features: Maximum number of features to extract.
        ngram_range: N-gram range for the vectorizer.

    Returns:
        Fitted TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    vectorizer.fit(texts)
    return vectorizer


def extract_text_features(texts: list[str], vectorizer: TfidfVectorizer) -> np.ndarray:
    """Extract text features using a fitted TF-IDF vectorizer.

    Args:
        texts: List of complaint text strings.
        vectorizer: Fitted TfidfVectorizer.

    Returns:
        Dense numpy array of shape (n_samples, max_features).
    """
    if not texts:
        n_feat = vectorizer.max_features or 300
        return np.empty((0, n_feat))
    features = vectorizer.transform(texts)
    return features.toarray()


def _one_hot(value: str, categories: list[str]) -> list[float]:
    """Create one-hot encoding for a categorical value."""
    return [1.0 if value == cat else 0.0 for cat in categories]


def extract_structured_features(df: pd.DataFrame) -> np.ndarray:
    """Extract structured features from a complaint DataFrame.

    Features extracted (in order):
    - location_zone: one-hot (5 dims)
    - time_of_day: one-hot (4 dims)
    - weather: one-hot (5 dims)
    - affected_people: normalized log
    - safety_risk: raw float
    - duplicate_count: derived from duplicate_group frequency
    - crowd_level: derived from affected_people (0/0.5/1)
    - vehicle_required: binary

    Total: 5 + 4 + 5 + 4 = 18 structured features.

    Args:
        df: Input DataFrame with complaint data.

    Returns:
        Numpy array of shape (n_samples, 18).
    """
    if df.empty:
        return np.empty((0, 18))

    # Pre-compute duplicate counts per group
    dup_counts = df['duplicate_group'].value_counts().to_dict() if 'duplicate_group' in df.columns else {}
    max_dup = max(dup_counts.values()) if dup_counts else 1

    features_list: list[list[float]] = []

    for _, row in df.iterrows():
        row_feat: list[float] = []

        # One-hot: location_zone (5)
        loc = str(row.get('location_zone', ''))
        row_feat.extend(_one_hot(loc, LOCATION_ZONES))

        # One-hot: time_of_day (4)
        tod = str(row.get('time_of_day', ''))
        row_feat.extend(_one_hot(tod, TIME_OF_DAY))

        # One-hot: weather (5)
        weather = str(row.get('weather', ''))
        row_feat.extend(_one_hot(weather, WEATHER_CONDITIONS))

        # Numeric: affected_people (log-normalized)
        affected = float(row.get('affected_people', 0))
        row_feat.append(np.log1p(affected) / 6.0)  # log1p(400)~6

        # Numeric: safety_risk
        row_feat.append(float(row.get('safety_risk', 0.0)))

        # Numeric: duplicate_count (normalized)
        dup_group = row.get('duplicate_group', -1)
        dup_count = dup_counts.get(dup_group, 1)
        row_feat.append(float(dup_count) / max(max_dup, 1))

        # Derived: crowd_level
        crowd_level = 1.0 if affected > 50 else (0.5 if affected > 10 else 0.0)
        row_feat.append(crowd_level)

        features_list.append(row_feat)

    return np.array(features_list, dtype=np.float32)


def combine_features(text_features: np.ndarray, structured_features: np.ndarray) -> np.ndarray:
    """Combine text and structured features horizontally.

    Args:
        text_features: Dense array of shape (n, text_dim).
        structured_features: Dense array of shape (n, struct_dim).

    Returns:
        Combined array of shape (n, text_dim + struct_dim).
    """
    if text_features.shape[0] == 0:
        return np.empty((0, text_features.shape[1] + structured_features.shape[1]))
    return np.hstack((text_features, structured_features))


def create_feature_pipeline(train_df: pd.DataFrame) -> dict[str, Any]:
    """Create a complete feature extraction pipeline from training data.

    Fits TF-IDF on complaint texts and creates category encoding mapping.

    Args:
        train_df: Training DataFrame with 'complaint_text' and 'category' columns.

    Returns:
        Pipeline dict with vectorizer, category_to_idx, and feature metadata.
    """
    texts = train_df['complaint_text'].fillna('').tolist()
    vectorizer = build_tfidf_vectorizer(texts, max_features=300, ngram_range=(1, 2))

    # Category encoding
    categories = sorted(train_df['category'].unique().tolist())
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    # Compute text feature dim
    sample_text_feat = extract_text_features(texts[:1], vectorizer)
    text_dim = sample_text_feat.shape[1]

    pipeline = {
        'vectorizer': vectorizer,
        'category_to_idx': cat_to_idx,
        'idx_to_category': {v: k for k, v in cat_to_idx.items()},
        'num_categories': len(categories),
        'text_feature_dim': text_dim,
        'structured_feature_dim': 18,
        'total_feature_dim': text_dim + 18,
        'feature_names': (
            [f'tfidf_{i}' for i in range(text_dim)] +
            [f'loc_{z}' for z in LOCATION_ZONES] +
            [f'tod_{t}' for t in TIME_OF_DAY] +
            [f'weather_{w}' for w in WEATHER_CONDITIONS] +
            ['affected_people_log', 'safety_risk', 'duplicate_count_norm', 'crowd_level']
        )
    }
    return pipeline


def transform_features(df: pd.DataFrame, pipeline: dict[str, Any]) -> np.ndarray:
    """Transform a DataFrame using a fitted feature pipeline.

    Args:
        df: DataFrame with 'complaint_text' and structured columns.
        pipeline: Fitted pipeline from create_feature_pipeline.

    Returns:
        Combined feature array of shape (n_samples, total_feature_dim).
    """
    texts = df['complaint_text'].fillna('').tolist()
    text_feat = extract_text_features(texts, pipeline['vectorizer'])
    struct_feat = extract_structured_features(df)

    return combine_features(text_feat, struct_feat)


def encode_categories(categories: pd.Series, pipeline: dict[str, Any]) -> np.ndarray:
    """Encode category labels to integer indices.

    Args:
        categories: Series of category strings.
        pipeline: Fitted pipeline with category_to_idx mapping.

    Returns:
        Numpy array of integer indices.
    """
    cat_to_idx = pipeline['category_to_idx']
    return np.array([cat_to_idx.get(c, 0) for c in categories], dtype=np.int64)
