# UserEngagement

PySpark pipeline that reads user engagement data and reports the average time spent per page and the most engaging page.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

```
python main.py
```

## Tests

```
pytest -q
```
