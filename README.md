

# Dataset Download Instructions (For Maintainers Only)

**Testers and users do NOT need to download the AG News dataset or set up a Kaggle API key.**

All required model artifacts will be provided in the repository (excluding raw data), so you can run, test, and use the app without any dataset download steps.

---

## For Maintainers (One-Time Setup)

If you need to re-generate the model artifacts from scratch, follow these steps:

### 1. Get your Kaggle API key

1. Go to https://www.kaggle.com/ → Sign in
2. Click on your profile picture (top right) → Account
3. Scroll down to the "API" section
4. Click "Create New API Token" or copy the API key provided
5. Copy your API key string (it will look like a long random string)

### 2. Set your API key as an environment variable

- **In VS Code Terminal (Windows PowerShell):**
	- Open a new terminal in VS Code (Terminal → New Terminal)
	- Paste and run:
		```powershell
		$env:KAGGLE_API_TOKEN="your_actual_api_key"
		```
- **Linux/Mac:**
	```bash
	export KAGGLE_API_TOKEN=your_actual_api_key
	```

You must run this command in the same terminal session before running the download script.

### 3. Download the dataset

Once your API key is set, run:

```
uv run python scripts/download_data.py
```

This will download and extract the AG News dataset into the `data/` directory.

**Note:** The `data/` directory is excluded from version control and will not be uploaded to GitHub.
