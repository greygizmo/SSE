
# GoSales MVP – Product Requirements Document  
*A multi‑product ICP & Whitespace Engine for Software Sales*

| Field | Value |
|-------|-------|
| **Version** | v0.1 (MVP) – 2025‑08‑03 |
| **Author** | ChatGPT (o3) |
| **Owner** | You (Product lead) |
| **Builder** | AI‑Coding Agent (Gemini CLI **or** OpenAI Codex) |
| **Audience** | Dev‑Agent + non‑SWE stakeholder |

---

## 0  Why This Exists (Plain English)

- Today our salespeople spray‑and‑pray—no data tells them **which accounts are most likely to buy each product**.  
- We want an engine that reads our full sales history and **scores every customer for every product line** (Simulation, PDM, Electrical, etc.).  
- It should also find **“whitespace”**—products a customer hasn’t bought but should.  
- This MVP must **run on a single laptop** and expose results in a simple Streamlit app.  
- I’m not a software engineer, so please code with explanatory comments, sensible defaults, and “guard rails” (_fail softly_, clear error logs).

---

## 1  Scope of the MVP

| Included | Notes |
|----------|-------|
| Ingest sample CSVs **and** connect to Azure SQL MI | Works even if DB creds are missing (fallback to SQLite). |
| Auto‑discover schema (no manual typing) | Generate YAML manifest for tables/columns. |
| Feature engineering per customer‑product | Spend, growth, seat CAGR, etc. |
| Two models per product | Logistic Regression (baseline) + LightGBM (higher accuracy). |
| Whitespace logic | Market Basket lift **+** ALS collaborative filtering. |
| Outputs | `outputs/icp_scores.csv`, `outputs/whitespace.csv`. |
| Streamlit UI | Search customer → see scores & whitespace. |
| Config via `.env` | Secrets never hard‑coded. |
| NO scheduling yet | Provide Prefect skeleton only. |
| NO auth in UI | Local use; keep simple. |

---

## 2  High‑Level Design Diagram

```text
CSV + Azure SQL ─► Ingest ► Star‑Schema ► Feature Views ─► ML Models
                                         ▲                  │
                                         └───── Scores ◄────┘
                                                │
                                            Streamlit
```

Human takeaway:

1. **Ingest** raw data.  
2. **Organize** into analytics‑friendly tables (star schema).  
3. **Calculate** features.  
4. **Train** models per product.  
5. **Score** customers + find whitespace.  
6. **Show** results in Streamlit.

---

## 3  Directory Layout (Best Practice)

```
gosales/
├─ .github/               # GH Actions later
├─ .devcontainer/         # (optional) VS Code remote dev
├─ data/                  # place raw CSVs here (git‑ignored)
├─ etl/                   # ingestion & schema discovery
│   ├─ load_csv.py
│   ├─ inspect_db.py
│   └─ build_star.py
├─ features/
│   └─ engine.py
├─ models/
│   ├─ train_simulation.py
│   ├─ train_pdm.py
│   └─ artifacts/         # *.pkl saved here (git‑ignored)
├─ whitespace/
│   ├─ build_lift.py
│   └─ als.py
├─ pipeline/
│   └─ score_all.py
├─ ui/
│   └─ app.py
├─ utils/
│   ├─ db.py              # SQLAlchemy engine helper
│   ├─ logger.py          # colored, timestamped logs
│   └─ paths.py           # central Path() helper
├─ tests/                 # pytest unit tests
├─ .env.template
├─ .gitignore
├─ .editorconfig
├─ ruff.toml              # linter config
├─ pyproject.toml         # deps + Black config
└─ README.md
```

### Why This Layout?

- **Separation of concerns** (ETL, features, models, UI).  
- Easy for an LLM to navigate; modules import via `utils`.  
- All outputs live in `/outputs` (git‑ignored) to keep repo clean.  
- Config/linting files help the coding agent follow PEP 8, Black, Ruff.

---

## 4  Key Files & What They Do

| File | Purpose | Teaching Note |
|------|---------|--------------|
| `etl/load_csv.py` | Load sample CSV into SQL table with dtype from stats CSV. | Shows agent how to cast numeric vs. text. |
| `etl/inspect_db.py` | Auto‑probe Azure SQL, spit YAML manifest. | Decouples us from unknown schemas. |
| `etl/build_star.py` | Build `fact_transactions`, `dim_customer`, `dim_product`. | Uses SQLAlchemy Core; okay if some columns missing. |
| `features/engine.py` | SQL → Pandas pipeline producing model matrix. | Each feature explained in docstrings. |
| `models/train_<product>.py` | Train baseline + LightGBM, save via MLflow. | Code picks best by AUC. |
| `whitespace/build_lift.py` | Apriori rules, outputs CSV of lifts. | Simple thresholds configurable. |
| `whitespace/als.py` | ALS matrix factorization for missing products. | Only runs if `implicit` installed. |
| `pipeline/score_all.py` | Joins features, loads model, outputs scores. | Single command to refresh metrics. |
| `ui/app.py` | Streamlit front‑end. | Startup banner = “GoSales MVP”. |

---

## 5  Model & Math Explainers (Human‑Readable)

1. **Logistic Regression**  
   - Predicts \( P(\text{buy}) \) using a weighted sum of features.  
   - Interpretable: a positive weight on “recent spend” means recent buyers are more likely to buy again.

2. **LightGBM**  
   - Gradient‑boosted trees—better accuracy on nonlinear patterns (e.g., interaction of industry × product mix).  
   - Still fast: 400 trees on 100 k rows runs in seconds locally.

3. **Market Basket Lift**  
   - If 30 % of customers who own CAD also own Simulation, but only 10 % of all customers own Simulation, lift = 3.  
   - High lift → “whitespace” product to pitch.

4. **ALS Collaborative Filtering**  
   - Treat orders as implicit ratings.  
   - Recommends products that “similar” customers bought.

These methods are industry‑standard; every parameter is commented in code so the agent—and you—see *why* they’re chosen.

---

## 6  Environment & Tooling

### 6.1  `requirements.txt`
```text
pandas
polars
numpy
sqlalchemy>=2.0
pyodbc
scikit-learn
lightgbm
mlflow
implicit          # ALS
mlxtend           # market basket
streamlit
prefect
python-dotenv
ruff
black
pytest
```

> **Tip**: the coding agent should create a virtualenv and run `pip install -r requirements.txt`.

### 6.2  `.env.template`
```env
AZSQL_SERVER=
AZSQL_DB=
AZSQL_USER=
AZSQL_PWD=
LLM_VENDOR=openai   # or gemini
OPENAI_API_KEY=
GEMINI_API_KEY=
```

The agent copies to `.env` for local dev. It falls back to SQLite if `AZSQL_*` is empty.

---

## 7  AI‑Coding Agent Guidance (`.aicoderules.md`)

```
# AI Coding Rules for GoSales

1. Treat warnings as non‑fatal; log and continue.
2. When schema unknown, call utils.db.explore() before hard‑coding column names.
3. All functions require type hints and docstrings.
4. Use utils.logger for prints—never bare print().
5. LLM calls must be wrapped in try/except; if key missing, return "unknown".
6. Respect .env for secrets.
7. Adhere to Black + Ruff on save.
8. Write unit tests for every utility function.
9. Commit messages: <scope>: <imperative summary>  (Conventional Commits).
```

The coding agent reads and obeys these rules.

---

## 8  Testing Strategy (For Non‑Engineers)

- **Unit tests** tell you each small function works (run `pytest -q`).  
- **End‑to‑end test** uses 100‑row sample CSV to ensure pipeline doesn’t crash.  
- **Data tests** (Great Expectations optional) check there’s a primary key and no null spend.

These tests run locally; green = safe to demo.

---

## 9  Running It All – Step‑by‑Step (For You)

```bash
# 1.  Clone the repo (agent will push to GitHub)
git clone <your-url> gosales && cd gosales

# 2.  Create a virtualenv
python -m venv .venv && source .venv/bin/activate

# 3.  Install dependencies
pip install -r requirements.txt

# 4.  Copy & fill environment vars
cp .env.template .env
#   (fill DB creds or leave blank for SQLite)

# 5.  Place CSV samples into data/
#    e.g. data/Sales_Log.csv

# 6.  Run the pipeline
make all         # or: python pipeline/score_all.py

# 7.  Launch UI
streamlit run ui/app.py
```

---

## 10  Milestone Timeline

| Day | Delivery Target |
|-----|-----------------|
| 1   | Repo scaffold & utils |
| 3   | CSV ingest + schema manifest |
| 6   | Star schema + features |
| 8   | Simulation model & scores |
| 10  | Whitespace modules |
| 12  | Full scoring run |
| 13  | Streamlit MVP |
| 15  | Unit tests + README |

---

## 11  Future Phases (Not in MVP)

- Auth (Azure AD)  
- Scheduled flows on Azure Functions  
- SHAP explanations in UI  
- MLOps retrain automation  
- Docker + CI/CD pipeline

---

### Final Note to Coding Agent

> **You have full autonomy** to choose idiomatic Python/SQL to accomplish these tasks. **Explain your reasoning in comments** so the product owner (non‑SWE) can trace logic.  
> Any unknown data column? Log a warning and default to `None`, don’t crash.  
> **End goal**: `streamlit run ui/app.py` shows an ICP score grid and whitespace list for any sample customer.

---

**End of PRD**.
