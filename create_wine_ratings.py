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
    rng = np.random.default_rng(random_seed)

    # Load dataset
    raw = pd.read_csv(path_vivno, encoding="utf-16", sep=";")
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
    w = split.iloc[:, :9].copy()
    w.columns = cols

    # Convert numeric columns
    w["Prices"] = (
        w["Prices"]
        .astype(str)
        .str.replace('"', "")
        .str.replace("$", "")
        .str.replace(",", "")
        .str.strip()
    )
    w["Prices"] = pd.to_numeric(w["Prices"], errors="coerce")
    w["ABV"] = pd.to_numeric(w["ABV"], errors="coerce")
    w["Ratings"] = pd.to_numeric(w["Ratings"], errors="coerce")
    w["Ratingsnum"] = (
        pd.to_numeric(w["Ratingsnum"], errors="coerce").fillna(0).astype(int)
    )

    # Categorize price into low/medium/high groups
    def price_level(p):
        if pd.isna(p):
            return "medium"
        if p < 20:
            return "low"
        elif p < 50:
            return "medium"
        else:
            return "high"

    # Categorize alcohol(ABV) into low/medium/high groups
    def abv_level(a):
        if pd.isna(a) or a < 5:
            return "medium"
        if a < 12:
            return "low"
        elif a < 14:
            return "medium"
        else:
            return "high"

    w["price_level"] = w["Prices"].apply(price_level)
    w["abv_level"] = w["ABV"].apply(abv_level)

    # Map regional names to primary country categories
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

    # Extract standardized country label from text
    def extract_country(x):
        if pd.isna(x):
            return "US"
        for key, val in country_map.items():
            if key.lower() in x.lower():
                return val
        return "US"

    w["country_norm"] = w["Countrys"].apply(extract_country)

    # Simplify wine color labels into broader categories
    def simplify_color(c):
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
        return "Red Wine"

    w["color_simple"] = w["color_wine"].apply(simplify_color)

    # Scale Ratingsnum to reduce dataset size
    w["Ratingsnum_scaled"] = (w["Ratingsnum"] * scale_factor).round().astype(int)

    # Create user personas
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

    # Select which users will rate each wine
    pairs = []
    for _, wine in w.iterrows():
        k = wine["Ratingsnum_scaled"]
        if k <= 0:
            continue

        sim = (
            (users["persona_country"] == wine["country_norm"]).astype(int)
            + (users["persona_color"] == wine["color_simple"]).astype(int)
            + (users["persona_price"] == wine["price_level"]).astype(int)
            + (users["persona_abv"] == wine["abv_level"]).astype(int)
        ) / 4.0

        weights = (sim + 0.01) / (sim + 0.01).sum()

        chosen = rng.choice(
            users["user_id"], size=min(k, n_users), replace=False, p=weights
        )

        for u in chosen:
            pairs.append([u, wine["Names"]])

    # Create DataFrame of user-item pairs and merge attributes
    df = pd.DataFrame(pairs, columns=["user_id", "item_id"])
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

    # Generate base rating using original wine rating + noise
    base = np.clip(
        np.round(df["Ratings"].fillna(3.0) + rng.normal(0, base_std, len(df))),
        1,
        5,
    ).astype(int)

    # Soft-weighted preference score
    df["rating"] = np.round(
        (1 - alpha) * base
        + alpha
        * (
            1
            + 4
            * (
                (df["persona_country"] == df["country_norm"]).astype(int)
                + (df["persona_color"] == df["color_simple"]).astype(int)
                + (df["persona_price"] == df["price_level"]).astype(int)
                + (df["persona_abv"] == df["abv_level"]).astype(int)
            )
            / 4.0
        )
    ).astype(int)

    # Final output DataFrame
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

    # Sort rows by user_id
    df_out["user_num"] = df_out["user_id"].str.extract(r"(\d+)").astype(int)
    df_out = (
        df_out.sort_values("user_num").drop(columns="user_num").reset_index(drop=True)
    )

    # Save dataset to CSV
    df_out.to_csv(path_out, index=False)
    print(f"Saved: {path_out} (with random_seed={random_seed})")
