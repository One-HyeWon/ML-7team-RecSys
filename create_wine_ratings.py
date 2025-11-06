"""
Generates a synthetic user-item rating dataset ('created_wine_ratings.csv')
based on the 'vivno_dataset.csv'.

This script creates user personas with specific wine preferences (country,
color, price, ABV) and generates ratings for wines based on a "soft-weighting"
system. This system combines the wine's original average rating (from
'vivno_dataset.csv') with a persona-wine similarity score, controlled
by an 'alpha' parameter.

The script ensures reproducibility by using a fixed 'random_seed' for all
random operations, which was a key requirement for the project.

Key Libraries:
- pandas: Used for data loading, manipulation, and merging (e.g., read_csv,
  merge, apply).
- numpy: Used for numerical operations and, crucially, for reproducible
  random number generation (RNG) via np.random.default_rng.
"""

import pandas as pd
import numpy as np


def create_wine_ratings(
    path_vivno="vivno_dataset.csv",
    path_out="created_wine_ratings.csv",
    n_users=200,
    alpha=0.35,
    base_std=0.60,
    scale_factor=0.01269,
    random_seed=2025,
):
    """
    Generates and saves a synthetic user-item rating dataset based on
    user personas and wine attributes.

    Parameters:
    ----------
    path_vivno : str
        File path to the raw vivno dataset (CSV format).
        (Default: "vivno_dataset.csv")
    path_out : str
        File path to save the generated ratings CSV file.
        (Default: "created_wine_ratings.csv")
    n_users : int
        The number of synthetic users to create. (Default: 200)
    alpha : float
        The weighting factor (0 to 1) for persona preference's influence
        on the final rating.
        - 0.0 = Ratings are based only on the wine's original score (plus noise).
        - 1.0 = Ratings are based only on persona-wine similarity.
        - 0.35 (default) = 35% persona score, 65% original score.
    base_std : float
        Standard deviation for the Gaussian (normal) noise added to the
        base rating (to create realistic variance). (Default: 0.60)
    scale_factor : float
        Factor to scale down 'Ratingsnum' (number of ratings) to
        reduce the final dataset size for performance. (Default: 0.01269)
    random_seed : int
        Seed for the NumPy random number generator to ensure reproducible
        results every time the script is run. (Default: 2025)

    Returns:
    -------
    None
        This function does not return a value but saves a CSV file to 'path_out'.
    """

    # --- 1. Reproducibility Setup ---

    # Initialize a modern NumPy Random Number Generator (RNG) with the seed.
    # This 'rng' object will be used for all random operations (choice, normal)
    # to ensure the *exact* same "random" results are produced every time.
    # (Explanation: np.random.default_rng is the recommended modern
    # replacement for np.random.RandomState, which was not taught in class.)
    rng = np.random.default_rng(random_seed)

    # --- 2. Data Loading and Initial Cleaning ---
    print(f"Loading raw data from {path_vivno}...")
    # Load the raw dataset.
    # (Explanation: 'encoding="utf-16", sep=";"' are used because the
    # provided CSV file is not in the standard utf-8, comma-separated format.)
    raw = pd.read_csv(path_vivno, encoding="utf-16", sep=";")

    # The raw data is loaded as a single messy column, so we split it by comma.
    split = raw.iloc[:, 0].str.split(",", expand=True)

    cols = [
        "Names",
        "color_wine",
        "Prices",
        "ML",
        "Ratings",
        "Ratingsnum",
        "Countrys",
        "ABV",
        "rates",
    ]
    # Select the first 9 columns from the split and assign correct names
    w = split.iloc[:, :9].copy()
    w.columns = cols

    # --- 3. Feature Engineering & Type Conversion ---

    # Convert 'Prices' column to numeric, removing non-numeric characters.
    w["Prices"] = (
        w["Prices"]
        .astype(str)
        .str.replace('"', "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    w["Prices"] = pd.to_numeric(w["Prices"], errors="coerce")

    # Convert 'ABV', 'Ratings', 'Ratingsnum' to numeric
    w["ABV"] = pd.to_numeric(w["ABV"], errors="coerce")
    w["Ratings"] = pd.to_numeric(w["Ratings"], errors="coerce")
    w["Ratingsnum"] = (
        pd.to_numeric(w["Ratingsnum"], errors="coerce").fillna(0).astype(int)
    )

    # --- 4. Helper Functions for Categorization ---
    # These functions are defined internally as they are only used here.

    def price_level(p):
        """Categorizes numeric price into 'low', 'medium', 'high'."""
        if pd.isna(p):
            return "medium"
        if p < 20:
            return "low"
        elif p < 50:
            return "medium"
        else:
            return "high"

    def abv_level(a):
        """Categorizes numeric ABV into 'low', 'medium', 'high'."""
        # [Critical Fix]: Treat 0.0 (and other low values) as missing data.
        # The original dataset uses 0.0 for unknown ABV. This was a
        # key problem discovered during EDA that this function solves.
        if pd.isna(a) or a < 5:
            return "medium"
        if a < 12:
            return "low"
        elif a < 14:
            return "medium"
        else:
            return "high"

    # Map regional names to primary country categories
    # This map is used to normalize messy 'Countrys' data (e.g., "Napa" -> "US")
    country_map = {
        "California": "US",
        "Napa": "US",
        "Sonoma": "US",
        "Oregon": "US",
        "Washington": "US",
        "Willamette": "US",
        "France": "France",
        "Bordeaux": "France",
        "Burgundy": "France",
        "Champagne": "France",
        "Rhône": "France",
        "Loire": "France",
        "Italy": "Italy",
        "Tuscany": "Italy",
        "Sicily": "Italy",
        "Piedmont": "Italy",
        "Spain": "Spain",
        "Rioja": "Spain",
        "Ribera": "Spain",
        "Germany": "Germany",
        "Australia": "Australia",
        "Portugal": "Portugal",
        "Argentina": "Argentina",
        "Chile": "Chile",
        "South Africa": "South Africa",
    }

    def extract_country(x):
        """Extracts a standardized country name from the 'Countrys' string."""
        if pd.isna(x):
            return "US"
        for key, val in country_map.items():
            if key.lower() in x.lower():
                return val
        return "US"  # Default to 'US' if no match

    def simplify_color(c):
        """Simplifies 'color_wine' text into 4 standard categories."""
        if pd.isna(c):
            return "Red Wine"
        if "Red" in c:
            return "Red Wine"
        if "White" in c:
            return "White Wine"
        if "Sparkling" in c or "Champagne" in c:
            return "Sparkling & Champagne"
        if "rose" in c or "rosé" in c or "pink" in c:
            return "Pink and Rosé"
        return "Red Wine"  # Default to 'Red Wine'

    # Apply the helper functions to create new categorical columns
    w["price_level"] = w["Prices"].apply(price_level)
    w["abv_level"] = w["ABV"].apply(abv_level)
    w["country_norm"] = w["Countrys"].apply(extract_country)
    w["color_simple"] = w["color_wine"].apply(simplify_color)

    # Scale 'Ratingsnum' (number of ratings) to control the size of
    # the final generated dataset.
    w["Ratingsnum_scaled"] = (w["Ratingsnum"] * scale_factor).round().astype(int)
    print("Feature engineering and categorization complete.")

    # --- 5. User Persona Generation ---

    # Create a DataFrame of 'n_users' with randomly assigned preferences.
    # We use 'rng.choice' (the RNG object's method) instead of
    # 'np.random.choice' to ensure this step is reproducible.
    # (Explanation: rng.choice(array, size, p=...) samples 'size' items
    # from 'array' with optional probabilities 'p'.)
    users = pd.DataFrame(
        {
            "user_id": [f"user_{i+1}" for i in range(n_users)],
            "persona_country": rng.choice(w["country_norm"].unique(), size=n_users),
            "persona_color": rng.choice(w["color_simple"].unique(), size=n_users),
            "persona_price": rng.choice(
                ["low", "medium", "high"], size=n_users, p=[0.3, 0.5, 0.2]
            ),
            "persona_abv": rng.choice(
                ["low", "medium", "high"], size=n_users, p=[0.25, 0.5, 0.25]
            ),
        }
    )
    print(f"Generated {n_users} user personas.")

    # --- 6. Rating Generation (Core Logic) ---

    print("Generating user-item rating pairs... (this may take a moment)")
    # Select which users will rate each wine
    pairs = []
    for _, wine in w.iterrows():
        # Use the scaled number of ratings
        k = wine["Ratingsnum_scaled"]
        if k <= 0:
            continue  # Skip wines with zero scaled ratings

        # Calculate similarity (0-1) between each user's persona and
        # the current wine's attributes (4 features).
        sim = (
            (users["persona_country"] == wine["country_norm"]).astype(int)
            + (users["persona_color"] == wine["color_simple"]).astype(int)
            + (users["persona_price"] == wine["price_level"]).astype(int)
            + (users["persona_abv"] == wine["abv_level"]).astype(int)
        ) / 4.0

        # Convert similarity scores into sampling probabilities.
        # Users with higher similarity are more likely to be chosen to rate this wine.
        # (sim + 0.01) ensures no zero-probability, preventing errors.
        weights = (sim + 0.01) / (sim + 0.01).sum()

        # Select 'k' users to rate this wine, weighted by persona similarity.
        # (Explanation: 'replace=False' ensures a user doesn't rate the
        # same wine twice in this step.)
        chosen = rng.choice(
            users["user_id"], size=min(k, n_users), replace=False, p=weights
        )

        for u in chosen:
            pairs.append([u, wine["Names"]])

    # Create the main DataFrame of (user_id, item_id) interactions
    df = pd.DataFrame(pairs, columns=["user_id", "item_id"])

    # Merge user personas and wine features into the interaction DataFrame
    df = df.merge(users, on="user_id", how="left").merge(
        w[
            [
                "Names",
                "country_norm",
                "color_simple",
                "price_level",
                "abv_level",
                "Ratings",
            ]
        ],
        left_on="item_id",
        right_on="Names",
        how="left",
    )

    # --- 7. Final Rating Calculation (Soft-Weighting) ---

    # 1. Generate base rating:
    # Take the wine's original 'Ratings' (or 3.0 if NaN).
    # Add Gaussian (normal) noise using the reproducible 'rng' object.
    # (Explanation: rng.normal(mean, std_dev, size) generates
    # 'size' random numbers from a normal distribution.)
    # 2. Clip:
    # (Explanation: np.clip(array, min, max) forces all values in
    # the array to be within the [min, max] range, i.e., 1 to 5 stars.)
    base = np.clip(
        np.round(df["Ratings"].fillna(3.0) + rng.normal(0, base_std, len(df))),
        1,
        5,
    ).astype(int)

    # 2. Generate the final soft-weighted rating.
    # This is the core logic of the rating generation:
    # final_rating = (1-alpha) * base_rating + (alpha) * persona_score
    # where persona_score is the 0-1 similarity score scaled to a 1-5 rating.
    df["rating"] = np.round(
        (1 - alpha) * base
        + alpha
        * (
            1  # Base score of 1 star
            + 4  # Additional 4 stars possible
            * (
                (df["persona_country"] == df["country_norm"]).astype(int)
                + (df["persona_color"] == df["color_simple"]).astype(int)
                + (df["persona_price"] == df["price_level"]).astype(int)
                + (df["persona_abv"] == df["abv_level"]).astype(int)
            )
            / 4.0  # Average similarity (0 to 1)
        )
    ).astype(int)

    # --- 8. Final Output ---

    # Select only the necessary columns for the final dataset
    df_out = df[
        [
            "user_id",
            "item_id",
            "rating",
            "persona_country",
            "persona_color",
            "persona_price",
            "persona_abv",
        ]
    ].copy()

    # Sort rows by user_id numerically (user_1, user_2, ... user_200)
    # (Explanation: .str.extract(r"(\d+)") pulls the number out of 'user_123')
    df_out["user_num"] = df_out["user_id"].str.extract(r"(\d+)").astype(int)
    df_out = (
        df_out.sort_values("user_num").drop(columns="user_num").reset_index(drop=True)
    )

    # Save the final dataset to the specified output path
    df_out.to_csv(path_out, index=False)
    print(f"\nSuccess! Saved: {path_out} (with random_seed={random_seed})")


# --- Script Execution ---
if __name__ == "__main__":
    # This line runs the main function defined above when the script is executed.
    # We explicitly pass 'random_seed=2025' to ensure that
    # 'created_wine_ratings.csv' is identical every time.
    print("Generating synthetic ratings dataset...")
    create_wine_ratings(
        path_vivno="vivno_dataset.csv",
        path_out="created_wine_ratings.csv",
        random_seed=2025,
    )
    print("Script finished.")
