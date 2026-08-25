from pathlib import Path

import numpy as np
import pandas as pd
import pygeohash as pgh
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder


DATA_DIR = Path("dataset")
BASE_FILE = Path("submission_kapow_killer.csv")
OUTPUT_FILE = Path("submission_rescue_killer_800_200.csv")

CAT_COLS = [
    "geohash",
    "geo5",
    "geo4",
    "geo3",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
]

LGB_SEEDS = [7, 13, 29, 43, 71]


def parse_time_slot(value: str) -> int:
    hour, minute = map(int, str(value).split(":"))
    return hour * 4 + minute // 15


def prepare_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train.drop(columns=["demand"]), test], ignore_index=True)
    combined["slot"] = combined["timestamp"].map(parse_time_slot)
    combined["hour"] = combined["slot"] // 4
    combined["minute"] = (combined["slot"] % 4) * 15

    for precision in (3, 4, 5):
        combined[f"geo{precision}"] = combined["geohash"].str[:precision]

    lat_lon = {geo: pgh.decode(geo) for geo in combined["geohash"].unique()}
    combined["lat"] = combined["geohash"].map(lambda geo: lat_lon[geo][0])
    combined["lon"] = combined["geohash"].map(lambda geo: lat_lon[geo][1])

    for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather"]:
        combined[col] = combined[col].fillna("Unknown").astype(str)

    combined["temp_missing"] = combined["Temperature"].isna().astype(int)
    combined["Temperature"] = combined.groupby(["geohash", "day"])["Temperature"].transform(
        lambda values: values.fillna(values.median())
    )
    combined["Temperature"] = combined["Temperature"].fillna(combined["Temperature"].median())

    prepared_train = combined.iloc[: len(train)].copy()
    prepared_test = combined.iloc[len(train) :].copy()
    prepared_train["demand"] = train["demand"].values
    return prepared_train, prepared_test


def add_group_mean(
    out: pd.DataFrame,
    source: pd.DataFrame,
    name: str,
    keys: list[str],
) -> None:
    means = source.groupby(keys)["demand"].mean()
    lookup_index = pd.MultiIndex.from_frame(out[keys])
    out[name] = means.reindex(lookup_index).to_numpy()


def add_previous_day_features(out: pd.DataFrame, day48: pd.DataFrame) -> None:
    add_group_mean(out, day48, "d48_geo_slot", ["geohash", "slot"])
    add_group_mean(out, day48, "d48_geo_hour", ["geohash", "hour"])
    add_group_mean(out, day48, "d48_g5_slot", ["geo5", "slot"])
    add_group_mean(out, day48, "d48_g4_slot", ["geo4", "slot"])
    add_group_mean(out, day48, "d48_g3_slot", ["geo3", "slot"])
    add_group_mean(out, day48, "d48_weather_slot", ["Weather", "slot"])
    add_group_mean(out, day48, "d48_road_slot", ["RoadType", "slot"])

    out["d48_geo_mean"] = out["geohash"].map(day48.groupby("geohash")["demand"].mean())
    out["d48_geo_max"] = out["geohash"].map(day48.groupby("geohash")["demand"].max())
    out["d48_slot_mean"] = out["slot"].map(day48.groupby("slot")["demand"].mean())
    out["d48_hour_mean"] = out["hour"].map(day48.groupby("hour")["demand"].mean())

    out["d48_base"] = (
        out["d48_geo_slot"]
        .fillna(out["d48_g5_slot"])
        .fillna(out["d48_g4_slot"])
        .fillna(out["d48_g3_slot"])
        .fillna(out["d48_weather_slot"])
        .fillna(0.65 * out["d48_geo_mean"] + 0.35 * out["d48_slot_mean"])
        .fillna(day48["demand"].mean())
    )


def add_early_day49_calibration(out: pd.DataFrame, observed: pd.DataFrame, day48: pd.DataFrame) -> None:
    observed = observed.sort_values(["geohash", "slot"]).copy()
    add_previous_day_features(observed, day48)

    paired = observed.dropna(subset=["d48_geo_slot"]).copy()
    global_delta = (paired["demand"] - paired["d48_geo_slot"]).mean()
    global_scale = (paired["demand"].sum() + 1e-6) / (paired["d48_geo_slot"].sum() + 1e-6)
    prior = 6

    for key in ["geohash", "geo5", "geo4", "geo3"]:
        stats = paired.groupby(key).agg(
            y_sum=("demand", "sum"),
            b_sum=("d48_geo_slot", "sum"),
            y_mean=("demand", "mean"),
            b_mean=("d48_geo_slot", "mean"),
            n=("demand", "size"),
        )
        stats["delta"] = (
            (stats["y_mean"] - stats["b_mean"]) * stats["n"] + prior * global_delta
        ) / (stats["n"] + prior)
        stats["scale"] = (stats["y_sum"] + prior * paired["demand"].mean()) / (
            stats["b_sum"] + prior * paired["d48_geo_slot"].mean()
        )

        out[f"{key}_delta"] = out[key].map(stats["delta"]).fillna(global_delta).clip(-0.12, 0.20)
        out[f"{key}_scale"] = out[key].map(stats["scale"]).fillna(global_scale).clip(0.35, 3.0)

        early_stats = observed.groupby(key).agg(
            early_mean=("demand", "mean"),
            early_max=("demand", "max"),
            early_last=("demand", "last"),
            early_n=("demand", "size"),
        )
        out[f"{key}_early_mean"] = out[key].map(early_stats["early_mean"]).fillna(
            observed["demand"].mean()
        )
        out[f"{key}_early_max"] = out[key].map(early_stats["early_max"]).fillna(
            observed["demand"].max()
        )
        out[f"{key}_early_last"] = out[key].map(early_stats["early_last"]).fillna(
            observed["demand"].mean()
        )
        out[f"{key}_early_n"] = out[key].map(early_stats["early_n"]).fillna(0)

    out["base_geo_delta"] = out["d48_base"] + out["geohash_delta"]
    out["base_geo_scale"] = out["d48_base"] * out["geohash_scale"]
    out["base_g5_delta"] = out["d48_base"] + out["geo5_delta"]
    out["base_g5_scale"] = out["d48_base"] * out["geo5_scale"]


def build_features(df: pd.DataFrame, observed: pd.DataFrame, day48: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    add_previous_day_features(out, day48)
    add_early_day49_calibration(out, observed, day48)

    out["time_sin"] = np.sin(2 * np.pi * out["slot"] / 96)
    out["time_cos"] = np.cos(2 * np.pi * out["slot"] / 96)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    excluded = set(["Index", "timestamp", "day", "demand", *CAT_COLS])
    return [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]


def encode_for_lgb(
    train_features: pd.DataFrame,
    predict_features: pd.DataFrame,
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_cat = pd.DataFrame(
        encoder.fit_transform(train_features[CAT_COLS]),
        columns=[f"cat_{col}" for col in CAT_COLS],
        index=train_features.index,
    )
    predict_cat = pd.DataFrame(
        encoder.transform(predict_features[CAT_COLS]),
        columns=[f"cat_{col}" for col in CAT_COLS],
        index=predict_features.index,
    )

    x_train = pd.concat([train_features[num_cols].fillna(-1), train_cat], axis=1)
    x_predict = pd.concat([predict_features[num_cols].fillna(-1), predict_cat], axis=1)
    return x_train, x_predict


def train_predict_blend(
    train_features: pd.DataFrame,
    predict_features: pd.DataFrame,
    y: np.ndarray,
    num_cols: list[str],
    validation: tuple[pd.DataFrame, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x_train, x_predict = encode_for_lgb(train_features, predict_features, num_cols)

    lgb_preds = []
    for seed in LGB_SEEDS:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=750,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        if validation is None:
            model.fit(x_train, y)
        else:
            val_features, val_y = validation
            _, x_val = encode_for_lgb(train_features, val_features, num_cols)
            model.fit(
                x_train,
                y,
                eval_set=[(x_val, val_y)],
                callbacks=[],
            )
        lgb_preds.append(model.predict(x_predict))

    x_cat_train = train_features[num_cols + CAT_COLS].copy()
    x_cat_predict = predict_features[num_cols + CAT_COLS].copy()
    for col in CAT_COLS:
        x_cat_train[col] = x_cat_train[col].astype(str)
        x_cat_predict[col] = x_cat_predict[col].astype(str)

    cat_model = CatBoostRegressor(
        iterations=850,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=5,
        loss_function="RMSE",
        random_seed=17,
        verbose=False,
    )
    cat_model.fit(
        x_cat_train,
        y,
        cat_features=[x_cat_train.columns.get_loc(col) for col in CAT_COLS],
    )
    cat_pred = cat_model.predict(x_cat_predict)

    formula_pred = np.clip(
        0.45 * predict_features["d48_base"].to_numpy()
        + 0.35 * predict_features["base_geo_scale"].to_numpy()
        + 0.20 * predict_features["base_geo_delta"].to_numpy(),
        0,
        1.2,
    )

    lgb_avg = np.mean(lgb_preds, axis=0)
    final_pred = 0.45 * lgb_avg + 0.40 * cat_pred + 0.15 * formula_pred
    final_pred = np.clip(final_pred, 0, 1.1)

    parts = {
        "lgb_avg": np.clip(lgb_avg, 0, 1.1),
        "cat": np.clip(cat_pred, 0, 1.1),
        "formula": formula_pred,
    }
    return final_pred, parts


def run_forward_validation(prepared_train: pd.DataFrame, day48: pd.DataFrame) -> None:
    day49 = prepared_train[prepared_train["day"] == 49].copy()
    observed = day49[day49["slot"] <= 5].copy()
    validation_rows = day49[(day49["slot"] >= 6) & (day49["slot"] <= 8)].copy()

    train_features = build_features(observed, observed, day48)
    val_features = build_features(validation_rows, observed, day48)
    num_cols = numeric_columns(train_features)

    preds, parts = train_predict_blend(
        train_features,
        val_features,
        observed["demand"].to_numpy(),
        num_cols,
    )
    y_true = validation_rows["demand"].to_numpy()

    print("\nForward validation: train day49 slots 0-5, validate slots 6-8")
    for name, pred in [("lgb_avg", parts["lgb_avg"]), ("cat", parts["cat"]), ("formula", parts["formula"]), ("blend", preds)]:
        rmse = mean_squared_error(y_true, pred) ** 0.5
        print(f"{name:8s} R2={r2_score(y_true, pred):.6f} RMSE={rmse:.6f} mean={pred.mean():.6f}")


def main() -> None:
    print("Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    prepared_train, prepared_test = prepare_frames(train, test)
    day48 = prepared_train[prepared_train["day"] == 48].copy()
    observed_day49 = prepared_train[prepared_train["day"] == 49].copy()

    print("Building forward validation...")
    run_forward_validation(prepared_train, day48)

    print("\nTraining final day49 transfer ensemble...")
    train_features = build_features(observed_day49, observed_day49, day48)
    test_features = build_features(prepared_test, observed_day49, day48)
    num_cols = numeric_columns(train_features)

    galaxy_pred, parts = train_predict_blend(
        train_features,
        test_features,
        observed_day49["demand"].to_numpy(),
        num_cols,
    )

    base = pd.read_csv(BASE_FILE)
    if not base["Index"].equals(test["Index"]):
        raise ValueError(f"{BASE_FILE} index order does not match test.csv")

    final_pred = np.clip(0.80 * base["demand"].to_numpy() + 0.20 * galaxy_pred, 0, 1.1)
    submission = pd.DataFrame({"Index": test["Index"], "demand": final_pred})
    submission.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {OUTPUT_FILE} with shape {submission.shape}.")
    print(submission["demand"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())
    print("\nComponent means:")
    print(f"lgb_avg={parts['lgb_avg'].mean():.6f} cat={parts['cat'].mean():.6f} formula={parts['formula'].mean():.6f}")


if __name__ == "__main__":
    main()
