# ECG HRV Feature Explorer

Setup:
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt

Run:
- streamlit run app/app.py

CSV:
- Columns: `ECG` (float), `Label` (0/1). Header case-insensitive.
- Default sampling rate: 750 Hz; configurable in the sidebar.
- Typical duration: 30s (22500 samples).
