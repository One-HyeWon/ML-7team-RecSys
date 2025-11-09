"""
Hybrid Wine Recommendation System - Part 2: Modeling & Evaluation

Project Overview:
This script loads the preprocessed data from Part 1 and builds the hybrid
recommendation system. It consists of two main models:

1.  A Collaborative Filtering (CF) model (SVD) trained on the
    `created_wine_ratings.csv` data. This model is effective for
    recommending items *within* the 'vivno' (Catalog A) dataset.
2.  A Content-Based Filtering (CBF) model (Random Forest) trained on
    "persona-item match features". This model is designed to solve the
    **Cold-Start** problem for the 'winemag' (Catalog B) dataset,
    recommending new wines by matching them to a user's learned persona.

Key Libraries Used (Not Taught in Class):
------------------------------------------
- surprise (Library):
    - Purpose: A Python library specifically designed for building and
      evaluating recommender systems.
    - `Reader(rating_scale=(1, 5))`: Defines the format and scale (1 to 5
      stars) of our rating data.
    - `Dataset.load_from_df(...)`: Loads a pandas DataFrame into a format
      that Surprise models can understand.
    - `surprise.model_selection.train_test_split`: A specific version
      of train_test_split for Surprise datasets.
    - `SVD` (Class): Implements the Singular Value Decomposition (SVD)
      algorithm, a common Matrix Factorization technique for CF.
    - `accuracy.rmse / .mae`: Functions to calculate Root Mean Squared
      Error and Mean Absolute Error, standard metrics for rating prediction.

- sklearn.ensemble.RandomForestClassifier:
    - `class_weight='balanced'`: This is the most critical parameter for
      our CBF model. Our ratings data (from EDA) was heavily biased
      towards 4-stars. This parameter automatically adjusts weights
      to give more importance to minority classes (like 3-star or
      5-star ratings), forcing the model to learn their patterns
      instead of just defaulting to the majority class.

- sklearn.metrics.ConfusionMatrixDisplay:
    - Purpose: A utility to plot a confusion matrix, which is essential
      for visualizing the performance of a classification model.
    - `from_predictions(..., normalize='true')`: We use `normalize='true'`
      to see the percentage (recall) of each class, which makes it
      much easier to spot bias (e.g., "80% of 5-star ratings were
      misclassified as 4-star").

- sys (Library):
    - Purpose: Provides access to system-specific parameters and functions.
    - `sys.argv`: A list of command-line arguments passed to the Python
      script. `sys.argv[0]` is the script name, and `sys.argv[1]` is the
      first argument (which we use to pass a `user_id`).

Sources:
- Datasets: Preprocessed CSV files from `preprocessing.ipynb`.
- Methodology: Model selection and "persona-item match" logic were
  developed iteratively based on evaluation results (e.g., fixing
  model bias and persona-mismatch issues).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# CF (Collaborative Filtering)
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split
from surprise import accuracy

# CBF (Content-Based Filtering)
from sklearn.model_selection import train_test_split as sklearn_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Ignore warning messages for cleaner output
import warnings

warnings.filterwarnings("ignore")

print("Modeling libraries loaded.")

# --- 0. Load Preprocessed Data ---

print("Loading preprocessed files...")

# 1. CF Training Data (User-Item-Rating)
ratings = pd.read_csv("created_wine_ratings.csv")

# 2. CBF Training Data (User-Item-Rating + Item Features + Persona Features)
merged_train_data = pd.read_csv("preprocessed_merged_train.csv")

# 3. Recommendation Target (Catalog B - 'winemag')
wm_clean = pd.read_csv("preprocessed_winemag.csv")

print(f"  - CF training data (ratings): {ratings.shape}")
print(f"  - CBF training data (merged_train_data): {merged_train_data.shape}")
print(f"  - Recommendation target (wm_clean): {wm_clean.shape}")

# --- 1. (Part 1) Collaborative Filtering (CF) Model ---
# Train and evaluate an SVD model on the 'vivno' catalog ratings.
# This model predicts ratings for (user, item) pairs *within* Catalog A.

print("\n[Part 1] Training and evaluating CF (SVD) model...")

# 1. Define the rating scale (1 to 5)
reader = Reader(rating_scale=(1, 5))
# Load data from DataFrame into Surprise's format
data = Dataset.load_from_df(ratings[["user_id", "item_id", "rating"]], reader)

# 2. Split the data into 80% training, 20% testing
cf_trainset, cf_testset = surprise_split(data, test_size=0.2, random_state=2025)

# 3. Initialize the SVD model
# (Explanation: n_factors=100 creates 100 latent features for users/items.
# random_state=2025 ensures reproducibility.)
model_svd = SVD(n_factors=100, n_epochs=30, random_state=2025)

# 4. Train the model
model_svd.fit(cf_trainset)

# 5. Evaluate the model on the test set
predictions = model_svd.test(cf_testset)
# (Explanation: accuracy.rmse calculates the Root Mean Squared Error)
cf_rmse = accuracy.rmse(predictions, verbose=False)  # verbose=False to silence output
cf_mae = accuracy.mae(predictions, verbose=False)

print("""\nCF Model Evaluation (on 'vivno' data):""")
print(f"  - RMSE: {cf_rmse:.4f}")
print(f"  - MAE: {cf_mae:.4f}")

# --- 2. (Part 2) Content-Based Filtering (CBF) Model ---
# Train a Random Forest model to predict ratings based *only* on the
# match between a user's persona and an item's features.
# This is the core of our solution for the 'winemag' (Catalog B) cold-start problem.

print("\n[Part 2] Training CBF (Random Forest) model based on 'Match' features...")

# 1. [KEY STRATEGY] Create "Match Features"
# Instead of feeding the model raw features (e.g., 'Spain', 'Chile'),
# we feed it the *result* of the comparison.
# This forces the model to learn the importance of a *match*.
merged_train_data["country_match"] = (
    merged_train_data["persona_country"] == merged_train_data["country_norm"]
).astype(int)
merged_train_data["color_match"] = (
    merged_train_data["persona_color"] == merged_train_data["color_simple"]
).astype(int)
merged_train_data["price_match"] = (
    merged_train_data["persona_price"] == merged_train_data["price_level"]
).astype(int)
merged_train_data["abv_match"] = (
    merged_train_data["persona_abv"] == merged_train_data["abv_level"]
).astype(int)

# 2. Define the features (X) and target (y) for the CBF model
# The model is trained *only* on these 4 match (0 or 1) features.
cbf_features = ["country_match", "color_match", "price_match", "abv_match"]
target = "rating"

X_cbf = merged_train_data[cbf_features]
y_cbf = merged_train_data[target]

# 3. Split the CBF data into 80% training, 20% testing
# (Explanation: stratify=y_cbf ensures that the 4-star bias
# is equally represented in both train and test sets.)
X_cbf_train, X_cbf_test, y_cbf_train, y_cbf_test = sklearn_split(
    X_cbf, y_cbf, test_size=0.2, random_state=2025, stratify=y_cbf
)

# 4. Initialize the Random Forest model
# (Explanation: No preprocessor (like OneHotEncoder) is needed
# because our features are already simple 0s and 1s.)
cbf_model = RandomForestClassifier(
    n_estimators=100,  # 100 decision trees
    random_state=2025,  # For reproducibility
    n_jobs=-1,  # Use all available CPU cores
    class_weight="balanced",  # [CRITICAL] Solves the 4-star rating bias
)

# 5. Train the CBF model
print("Training CBF model on 'Match' features...")
cbf_model.fit(X_cbf_train, y_cbf_train)

print("CBF model training complete.")


# --- 3. (Part 3) CBF Model Evaluation ---
# Evaluate how well the CBF model learned to classify ratings.

print("\n[Part 3] Evaluating CBF model...")

# 1. Get predictions on the test set
y_cbf_pred = cbf_model.predict(X_cbf_test)

# 2. Print Classification Report
# (Explanation: This shows Precision, Recall, and F1-score for each
# rating class. We expect low accuracy (due to 'balanced' weights)
# but better recall for minority classes (3 and 5).)
print("--- CBF Model Classification Report ---")
report = classification_report(y_cbf_test, y_cbf_pred)
print(report)

# 3. Plot Confusion Matrix
print("--- CBF Model Confusion Matrix ---")
fig, ax = plt.subplots(figsize=(8, 6))
# (Explanation: ConfusionMatrixDisplay.from_predictions plots
# True Label vs. Predicted Label. normalize='true' shows the
# percentage (recall) for each True Label row.)
ConfusionMatrixDisplay.from_predictions(
    y_cbf_test, y_cbf_pred, ax=ax, cmap="Blues", normalize="true"
)
ax.set_title("CBF Model Confusion Matrix (Normalized)")
plt.show()


# --- 4. (Part 4) Hybrid Recommendation Function ---
# This function defines the core logic for recommending from Catalog B.

print("\n[Part 4] Defining hybrid recommendation function...")


def get_hybrid_recommendations(user_persona, n_recs=10):
    """
    Generates Top-N wine recommendations from 'winemag' (Catalog B)
    by creating "Match Features" on-the-fly for a given user persona.

    Parameters:
    - user_persona (pd.Series): A single row from the 'ratings' DataFrame
      containing the user's preferences (e.g., 'persona_country').
    - n_recs (int): The number of recommendations to return.

    Returns:
    - pd.DataFrame: A DataFrame of the Top-N recommended wines.
    """

    # 1. Get all 130k candidate items from Catalog B
    X_candidates = wm_clean.copy()

    # 2. [KEY STRATEGY] Create the 4 "Match Features" in real-time
    # We compare the single user's persona against all 130k items.
    X_candidates["country_match"] = (
        X_candidates["country_norm"] == user_persona["persona_country"]
    ).astype(int)
    X_candidates["color_match"] = (
        X_candidates["color_simple"] == user_persona["persona_color"]
    ).astype(int)
    X_candidates["price_match"] = (
        X_candidates["price_level"] == user_persona["persona_price"]
    ).astype(int)
    X_candidates["abv_match"] = (
        X_candidates["abv_level"] == user_persona["persona_abv"]
    ).astype(int)

    # 3. Select only the features the model was trained on
    X_candidates_features = X_candidates[cbf_features]

    # 4. Get rating *probabilities* for all 130k items
    print(
        f"Predicting scores for {len(X_candidates)} items based on "
        f"Persona ({user_persona['persona_country']}/{user_persona['persona_price']})..."
    )
    # (Explanation: .predict_proba() returns an array where each row
    # is a list of probabilities for each class, e.g., [P(2), P(3), P(4), P(5)])
    pred_proba = cbf_model.predict_proba(X_candidates_features)

    # 5. Calculate a "Positive Score"
    # We define a "good" recommendation as a high probability
    # of being 4-stars OR 5-stars.
    classes = cbf_model.classes_
    score_cols = [i for i, r in enumerate(classes) if r >= 4]

    if not score_cols:
        print("Warning: Model did not find classes 4 or 5. Defaulting to 5.")
        score_cols = [i for i, r in enumerate(classes) if r == 5]

    positive_score = pred_proba[:, score_cols].sum(axis=1)

    # 6. Add the score to the results and sort
    wm_results = wm_clean.copy()
    wm_results["recommend_score"] = positive_score

    # 7. Get the Top-N highest-scoring recommendations
    top_n_recs = wm_results.sort_values(by="recommend_score", ascending=False).head(
        n_recs
    )

    cols_to_show = [
        "title",
        "variety",
        "country_norm",
        "price",
        "points",
        "recommend_score",
        "color_simple",
        "price_level",
    ]

    return top_n_recs[cols_to_show]


print("Recommendation function 'get_hybrid_recommendations' defined.")


# --- 5. (Part 5) Analysis Function ---
# A helper function to run and visualize the recommendation for one user.


def analyze_user_recommendations(user_id="user_20", n_recs=10):
    """
    Runs the end-to-end analysis for a single user ID:
    1. Finds the user's persona.
    2. Calls get_hybrid_recommendations().
    3. Prints the results.
    4. Plots a 3-part chart comparing the persona to the results.

    Parameters:
    - user_id (str): The ID of the user to analyze (default: 'user_20').
    - n_recs (int): The number of recommendations to generate (default: 10).
    """

    print(f"\n[Part 5] Starting analysis for user: '{user_id}'...")

    # 1. Find the user's persona from the 'ratings' DataFrame
    try:
        user_persona = ratings[ratings["user_id"] == user_id].iloc[0]
    except IndexError:
        # Handle cases where the user_id is not found
        print(f"--- 🚫 ERROR: User ID '{user_id}' not found in 'ratings'. ---")
        return

    print(f"--- 👤 {user_id} Persona Profile ---")
    print(f"  - Prefers Country: {user_persona['persona_country']}")
    print(f"  - Prefers Color:   {user_persona['persona_color']}")
    print(f"  - Prefers Price:   {user_persona['persona_price']}")
    print(f"  - Prefers ABV:     {user_persona['persona_abv']}")
    print("---------------------------------")

    # 2. Generate recommendations for this persona
    print(f"\nGenerating Top {n_recs} recommendations for '{user_id}'...")
    recommendations = get_hybrid_recommendations(user_persona, n_recs=n_recs)

    # 3. Print the final list to the console
    print(f"\n--- 🍷 Top {n_recs} Recommendations for {user_id} ---")
    print(recommendations)

    # 4. Plot the qualitative analysis
    print("\n--- 🔬 Qualitative Analysis ---")
    print(f"Comparing {n_recs} recommendations against the user's persona...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Analysis: '{user_id}' Persona vs. Recommendations", fontsize=16, y=1.05
    )

    # Plot 1: Country
    sns.countplot(
        ax=axes[0],
        data=recommendations,
        y="country_norm",
        order=recommendations["country_norm"].value_counts().index,
        palette="crest",
    )
    axes[0].set_title(f"Country (Persona: {user_persona['persona_country']})")

    # Plot 2: Color
    sns.countplot(
        ax=axes[1],
        data=recommendations,
        x="color_simple",
        order=recommendations["color_simple"].value_counts().index,
        palette="flare",
    )
    axes[1].set_title(f"Color (Persona: {user_persona['persona_color']})")

    # Plot 3: Price Level
    sns.countplot(
        ax=axes[2],
        data=recommendations,
        x="price_level",
        order=["low", "medium", "high"],
        palette="mako",
    )
    axes[2].set_title(f"Price Level (Persona: {user_persona['persona_price']})")

    plt.tight_layout()
    plt.show()


# --- 6. (Part 6) Script Execution (Entry Point) ---
# (Explanation: `if __name__ == "__main__":` is a standard Python
# convention. This block of code will *only* run when the script is
# executed directly (e.g., `python model.py`), not when it is
# imported as a module by another script.)
if __name__ == "__main__":

    # (Explanation: `sys.argv` is the list of command-line arguments.
    # `len(sys.argv) > 1` checks if the user provided an argument
    # (e.g., `python model.py user_30`).)
    if len(sys.argv) > 1:
        # If an argument is given, use it as the user_id
        user_to_analyze = sys.argv[1]
        analyze_user_recommendations(user_id=user_to_analyze)
    else:
        # If no argument is given, run the analysis for the default user
        print("No user_id provided. Running default analysis for 'user_20'.")
        analyze_user_recommendations(user_id="user_20")
