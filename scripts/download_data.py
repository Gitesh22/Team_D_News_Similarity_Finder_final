import os
from pathlib import Path


def download_ag_news():
    # Check for Kaggle API token in environment
    kaggle_token = os.environ.get("KAGGLE_API_TOKEN")
    if not kaggle_token:
        raise OSError(
            "KAGGLE_API_TOKEN environment variable not set. "
            "Please set it before running this script."
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("Please install the 'kaggle' package: uv sync kaggle") from None

    api = KaggleApi()
    api.authenticate()
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)
    api.dataset_download_files(
        "amananandrai/ag-news-classification-dataset", path=str(data_dir), unzip=True
    )
    print(f"Dataset downloaded and extracted to: {data_dir}")


if __name__ == "__main__":
    download_ag_news()
