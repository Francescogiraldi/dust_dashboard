# Dust Dashboard (Streamlit)

A Streamlit dashboard to explore Dust workspace usage data with a fully French UI and configurable metrics.

## Requirements
- Python 3.9+ (recommended 3.10+)
- `pip`

## Installation
```bash
pip install -r requirements.txt
```

## Secrets
You can provide your API key in one of two ways:
- Enter it in the sidebar under "Clé API (Bearer)" (masked).
- Or create `.streamlit/secrets.toml` and add:
  ```toml
  DUST_API_KEY = "<your_api_key>"
  ```
`.streamlit/` is excluded by `.gitignore` so secrets are not committed.

## Run
```bash
streamlit run app.py
```
Open the app at `http://localhost:8501/`.

## Usage
- Sidebar configuration:
  - `Région API`: choose the Dust region (e.g., `US (us-central1)`).
  - `ID du workspace`: your workspace ID.
  - `Clé API (Bearer)`: API key (masked).
  - `Mode`: `Mois` or `Plage de dates`.
  - `Table`: `Utilisateurs`, `Messages d’assistants`, `Builders`, `Assistants`, `Feedbacks`, or `Tous (ZIP)`.
  - `Format`: `JSON` or `CSV`.
  - `Inclure les inactifs (0 messages)`: include entries with 0 messages.
  - `Préférer les noms (pas d’ID)`: display names when available.
  - Section "Affichage des métriques": toggle visibility of charts/tables:
    - Volumes par jour
    - Volumes par assistant par jour
    - Top utilisateurs
    - Top assistants
    - Feedbacks
- Click `Récupérer les données` to fetch and render charts/tables.

## Troubleshooting
- 403 (Accès refusé): Ask the admin to enable "Workspace Usage API" for your workspace on `https://dust.tt` or contact support.
- Other errors: Open the `Debug requête API` panel in the UI and share the `status` and body content for diagnosis.

## Notes
- When `Table = Tous (ZIP)` with `Format = CSV`, the app loads multiple tables from a ZIP; name lookups are applied when possible.
- With single tables (e.g., `assistant_messages`), the app tries to map IDs to names using available columns or background lookups.