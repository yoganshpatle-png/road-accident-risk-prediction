"""Run the full pipeline: data -> clean -> features -> train -> evaluate -> hotspots -> map."""
from src.config import OUTPUT_DIR
from src.data_collection import load_data
from src.evaluate import evaluate_model
from src.hotspot_analysis import find_hotspots
from src.preprocessing import clean_data, save_processed
from src.train_model import train_and_select


def main():
    raw = load_data()
    df = clean_data(raw)
    save_processed(df)

    best, table, (X_test, y_test) = train_and_select(df)
    print("\nModel comparison:\n", table.to_string(index=False))
    metrics = evaluate_model(best, X_test, y_test)
    print("\nBest model metrics:", {k: round(v, 3) for k, v in metrics.items()})

    clustered, hotspots = find_hotspots(df)
    hotspots.to_csv(OUTPUT_DIR / "hotspots.csv", index=False)
    print(f"\n[hotspots] {len(hotspots)} hotspots found\n", hotspots.head(10).to_string(index=False))

    try:
        from src.map_visualization import build_map, save_map
        path = save_map(build_map(clustered, hotspots))
        print(f"\n[map] saved to {path}")
    except ImportError:
        print("\n[map] folium not installed, skipping map (pip install folium)")


if __name__ == "__main__":
    main()
