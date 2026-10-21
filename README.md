# 3A Lead Agent

Finds new client prospects for **3A Composites Core Materials** (AIREX, BALTEK, LANTOR, ENGICORE).

It starts from 10 seed companies, learns what makes them good clients, searches the web for
companies like them, scores every match, and lets you accept or reject each lead in a small web
app. Your decisions and comments are fed back into the profile, so every new search gets better.

```
1 Taxonomy -> 2 Seed profiles -> 3 Normalizer -> 4 Product fit -> 5 Ideal Client Profile
  -> 6 Discover -> 7 Match + rank -> 8 Review (web app) -> 9 Learn -> back to 6
```

| Stage | What it does | Engine |
|---|---|---|
| 1 Taxonomy | Builds the 3A product dictionary from 3accorematerials.com and lantorcomposites.com | Exa Agent (ultra) |
| 2 Seed profiles | One manufacturing profile per seed company | Exa Agent (ultra) |
| 3 Normalizer | Turns research text into comparable tokens such as `PET`, `VACUUM_INFUSION` | Rules, LLM for leftovers |
| 4 Product fit | Which 3A products fit each company (`current_stack` or `need_bridge`) | LLM |
| 5 Ideal Client Profile | Merges all seeds into one "wanted poster" for the search | Merge logic + LLM |
| 6 Discover | Finds new companies that match the profile | Exa Agent (list-build) |
| 7 Match + rank | Product fits and a score for every new company | Python |
| 8 Review | Accept or reject each lead, with an optional comment | FastAPI web app |
| 9 Learn | Turns your decisions into rules and a new profile version | LLM |


## 1. One-time setup

You need **Python 3.10 or newer**. Check with `python3 --version`.
(On a Mac without it: install it from [python.org](https://www.python.org/downloads/) or with `brew install python`.)

Open the Terminal app and run:

```bash
cd ~/"3a lead agent"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**API keys.** They live in the `.env` file in this folder (it is already filled in for you).
To start from scratch, copy `.env.example` to `.env` and add your keys. `.env` holds secrets: never share it.

| Setting | Used for |
|---|---|
| `EXA_API_KEY` | Web research (stages 1, 2 and 6) |
| `OPENAI_API_KEY` | The LLM (when `LLM_PROVIDER=openai`) and the text similarity used for ranking |
| `LLM_PROVIDER` | `openai` (default) or `anthropic` |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` | The model to use for each provider |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID` | Only for `LLM_PROVIDER=anthropic`. The workspace id is needed when Anthropic says the key "is not scoped to a workspace" |


## 2. Run it

Every time you open a new Terminal window, first go to the folder and switch on the environment:

```bash
cd ~/"3a lead agent"
source .venv/bin/activate
```

**Step 1: check your keys** (cheap test calls, a few seconds):

```bash
python -m lead_agent check
```

**Step 2: build the profile and find the first leads** (stages 1 to 7):

```bash
python -m lead_agent run
```

This is the long step: the first run usually takes one to three hours, mostly waiting for Exa.
Each "ultra" research run takes up to 30-60 minutes, and five seed companies are researched at a
time. With the default budgets the first full run costs at most about $50 in Exa usage
(taxonomy $10 + 10 seeds x $3 + one search $10), often less. LLM costs are small.
You can stop it with Ctrl+C at any time: run the same command again and it continues where it
left off, including Exa runs that were still in progress.

**Step 3: review the leads:**

```bash
python -m lead_agent review
```

Then open **http://127.0.0.1:8000** in your browser. For each lead you see why it was suggested,
which 3A products fit, the signals that were found, the sources, and the score.

- **Accept** or **Reject** it (keyboard: `A` / `R`, and `J` / `K` to move through the list).
  A short comment on why helps the agent learn ("builds aluminium boats only").
- **Learn from decisions** turns your decisions into rules and a new Ideal Client Profile version.
- **Find new leads** runs a new search with the latest profile, in the background. New decisions
  are always learned first.

Stop the web app with Ctrl+C in the Terminal.


## 3. All commands

| Command | What it does |
|---|---|
| `python -m lead_agent check` | Test the API keys |
| `python -m lead_agent run` | Stages 1-7: build whatever is missing, then find and rank new leads |
| `python -m lead_agent review` | Stage 8: the review web app (`--port 8001` to use another port) |
| `python -m lead_agent learn` | Stage 9: update the profile from your decisions |
| `python -m lead_agent discover` | Stages 6-7: find and rank more leads (learns new decisions first) |
| `python -m lead_agent discover --run RUN_ID` | Rank an earlier search again without paying for a new one (if ranking failed) |
| `python -m lead_agent taxonomy` | Stage 1 only |
| `python -m lead_agent seeds` | Stage 2 only |
| `python -m lead_agent normalize` | Stage 3 only |
| `python -m lead_agent fit` | Stage 4 only |
| `python -m lead_agent profile` | Stage 5 only |

Stages skip work that is already done, and redo it when its inputs changed. So after you edit
`inputs/seed_companies.csv` or `inputs/normalization_rules.json`, just run `python -m lead_agent run`
again: new seeds are researched, and the tokens, product fits and Ideal Client Profile are rebuilt
where needed. Add `--force` to redo a stage anyway (for example `seeds --force` researches every
seed again, which costs Exa credits). `taxonomy --force` and a rebuilt profile are saved as new
versions; a rebuilt profile learns from all your past review decisions again on the next search.


## 4. Folder structure

```
3a lead agent/
├── README.md
├── requirements.txt
├── .env                  your API keys (secret)
├── .env.example          template for .env
├── inputs/
│   ├── seed_companies.csv        the 10 seed companies (company, website)
│   └── normalization_rules.json  the token dictionary used by Stage 3
├── prompts/              the text sent to Exa and the LLM, one file per stage (edit freely)
├── lead_agent/           the program
│   ├── __main__.py       the commands above
│   ├── pipeline.py       runs the stages in order
│   ├── settings.py       budgets, models, score weights: all tunable values
│   ├── storage.py        where every file is saved
│   ├── check.py          the key check
│   ├── errors.py, utils.py, prompts.py
│   ├── clients/          Exa Agent, LLM (OpenAI or Anthropic), embeddings, HTTP retries
│   └── stages/           one module per stage: stage1_taxonomy.py ... stage9_learning.py
│       └── stage8_review/    the web app: app.py (FastAPI), service.py, static/index.html
├── tests/                offline tests with fake Exa and LLM answers
└── output/               everything the pipeline produces (created on the first run)
```

Everything the pipeline produces is plain JSON in `output/`, following the plan's artifact map:

```
output/data/taxonomy_v1.json                         Stage 1
output/profiles/{seed_id}/manufacturer_profile.json  Stage 2
output/data/taxonomy_v1_normalized.json              Stage 3 (raw + normalized)
output/profiles/{seed_id}/normalized.json            Stage 3 (raw + normalized)
output/profiles/{seed_id}/product_fit.json           Stage 4
output/data/ideal_client_profile_v1.json             Stage 5 (v2, v3... written by Stage 9)
output/runs/{run_id}/prospects.json                  Stage 6
output/leads/{lead_id}.json                          Stage 7
output/decisions/{lead_id}.json                      Stage 8
output/data/disliked_examples.json                   Stage 9 (rejected leads, for the ranking penalty)
```


## 5. How leads are scored

```
score = need_overlap + stack_overlap + profile_similarity + exa_confidence - dislike_similarity
```

| Part | Meaning (each part is between 0 and 1) |
|---|---|
| `need_overlap` | How much of the ideal profile's needs the company shows, weighted by how many seeds share each need |
| `stack_overlap` | The same for materials and processes |
| `profile_similarity` | Text similarity between the company and the profile's description (OpenAI embeddings) |
| `exa_confidence` | Exa's match confidence: high 1.0, medium 0.6, low 0.3 |
| `dislike_similarity` | How much closer the company is to a lead you rejected than to the ideal profile (0 when it is not) |

The weights are in `SCORE_WEIGHTS` in `lead_agent/settings.py`.


## 6. Things you can tune

- **Cost and depth of Exa research:** `EXA_RUNS` in `lead_agent/settings.py` (effort, maximum dollars and
  minutes per run). `auto` instead of `ultra` is faster and cheaper; a fixed effort such as `high` has a
  fixed price per run.
- **Leads per search:** `PROSPECTS_PER_RUN` in `lead_agent/settings.py`.
- **What Exa and the LLM are asked:** the files in `prompts/`.
- **Token rules:** `inputs/normalization_rules.json`. Patterns match whole words, ignore case and allow
  a plural "s". End a pattern with `*` to match the start of a word: `recyclab*` matches "recyclable"
  and "recyclability" (this is the one change made to the original file). Each attribute only uses the
  rule categories that fit it, so a list of core materials can only produce core tokens.


## 7. Tests

The tests run the whole loop offline with fake Exa and LLM answers (no cost, about a second):

```bash
python -m unittest
```


## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `... is not set. Add it to the .env file` | Add the missing key to `.env` |
| The LLM check fails with a model error | Set `OPENAI_MODEL` in `.env` to a model your OpenAI account can use |
| Anthropic says the key "is not scoped to a workspace" | Add `ANTHROPIC_WORKSPACE_ID` to `.env`, or keep `LLM_PROVIDER=openai` |
| A run was interrupted | Run the same command again: finished work is kept and Exa runs in progress are resumed (they are listed in `output/exa_pending/`) |
| `Address already in use` when starting the review app | Another copy is running, or use `python -m lead_agent review --port 8001` |
| Start completely fresh | Delete the `output/` folder |
