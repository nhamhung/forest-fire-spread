# Forest Fire Spread: Monte Carlo Simulation and El Niño

Monte Carlo wildfire spread on a grid. The project asks:

* how forest density, drought and wind control fire size;
* where the critical density lies at which fires switch from burning out to crossing the whole forest;
* which fire-prevention strategy saves the most forest.

It then tests the El Niño connection against 25 years of NASA satellite fire data for Indonesia and Australia, with
the 2026–27 El Niño as context.

| Phase | Model | Key result |
|---|---|---|
| 1 | Deterministic 4-neighbour fire (site percolation) | Critical density 0.5928 (exact 0.5927); transition sharpens as L^−0.76 (theory −0.75) |
| 2 | Spread probability q; 8-neighbour option | Wetter fuel raises the threshold: 0.593 (q = 1) → 0.698 (q = 0.8) → 0.867 (q = 0.6); three more known thresholds reproduced |
| 3 | Drought (dryness) and wind | Critical dryness 0.96 / 0.70 / 0.50 / 0.16 for density 0.60 / 0.65 / 0.70 / 0.80; 10 m/s wind makes the head 2.3× faster |
| 4 | Firebreaks, thinning, controlled burns; ember spotting | Firebreaks save 21 trees per tree removed at a 4% budget (thinning 2.9, controlled burns 1.3); strong embers breach widely spaced breaks |
| 5 | NASA FIRMS fire detections vs NOAA Niño 3.4 | Indonesian El Niño fire seasons average 3.1× neutral ones (ρ = 0.73); Australia 1.5× (ρ = 0.51); Niño 3.4 +2.09 °C in Sep 2026 |
| 6 | Streamlit app | Live fire spread with sliders for density, spread chance, dryness, wind, embers and management, plus side-by-side comparison |

![Phase 4: the same forest and lightning strike under each fuel treatment](results/phase4_examples/figure.png)

## Reports

* [Project report](https://nhamhhung.github.io/forest-fire-spread/) (*When Forests Catch Fire*): methods, results, the El Niño data
  comparison and limitations. Every number is read from `results/`.
* [`report/user_guide.html`](report/user_guide.html) (*Using the Forest Fire Simulator*): a guided tour of the app
  with screenshots, three hands-on experiments, troubleshooting and a glossary.

## The app

```bash
streamlit run app/streamlit_app.py      # from the project root, so the theme in .streamlit/ applies
```

* **Sidebar:** one-click scenarios (Normal year, El Niño drought, Drought + wind, Drought + firebreaks), forest
  density and size, 4 or 8 neighbours, normal-year spread chance, dryness, wind speed and direction, ignition,
  management strategy and budget. Under *Advanced*: burn duration, ember spotting and seed.
* **SIMULATE:** a verdict comparing your forest with the critical density for this year's fuel, the outcome, and
  an animated fire (play / pause / scrub) with a burned-over-time chart. *Compare side by side* burns the same forest
  and spark with and without management.
* **EXPERIMENTS:** the saved figures from every phase.
* **EL NIÑO DATA:** monthly fire detections and Niño 3.4, one dot per fire season, and statistics by ENSO phase.
* **Export:** a standalone HTML replay and the scenario as JSON.

## The physics in brief

* **Phase 1.** Each cell is a tree with probability p; the left edge is lit; fire spreads to side-neighbouring trees.
  Whether it crosses is site percolation, with a sharp threshold at p_c ≈ 0.5927.
* **Phase 2.** Each spread attempt succeeds with probability q (lower = wetter fuel), so a denser forest is needed.
  Below q = 1/2 no forest carries fire. Diagonal spread lowers every threshold (0.407 for dry fuel).
* **Phase 3.**
  * *Dryness d* (El Niño drought) sets q = 1 − (1 − q_normal)(1 − d).
  * *Wind* multiplies spread chances by e^(0.131·V·cos θ), the angular coefficient of Alexandridis et al. (2008).
  * For wind runs, trees smoulder for 10 steps, so wind can change speed and not only direction.
* **Phase 4.** Trees are removed before the season by:
  * firebreaks (a grid of cleared lines);
  * random thinning;
  * controlled-burn patches.

  One random lightning strike follows. Breaking connectivity beats removing fuel. Thinning works only once it
  pushes the forest below the critical density. Optional ember spotting lets fire jump gaps.
* **Phase 5.** NASA FIRMS MODIS detections (confidence ≥ 30, vegetation fires), 2001–2025, per fire season
  (Indonesia Aug–Nov, Australia Sep–Dec), against the NOAA Niño 3.4 anomaly over the same months.

**Limits:**

* This is a teaching model: no terrain, fuel types, peat, suppression or full fire weather.
* The dryness and wind mappings are illustrative, and without embers a firebreak is a perfect wall.
* The real-data comparison is a correlation, not a forecast or attribution. FIRMS had not yet published 2026.

## Layout

```
src/forest_fire/
  grid.py          # cell states, random forest, left-edge / point ignition
  spread.py        # spread rule (Numba): per-direction probabilities, 4 or 8 neighbours, burn duration, embers
  weather.py       # drought (dryness -> q) and wind (direction-dependent spread)
  management.py    # firebreaks, random thinning, controlled burns
  experiments.py   # metrics, sweeps, logistic fits, critical density / dryness, wind shapes, management comparison
  real_data.py     # FIRMS + Niño 3.4 parsing, seasonal tables, phase comparison, permutation test
  plotting.py      # maps, sweep plots, phase diagram, animations
  results.py       # save_result: settings.json (+ versions) next to data.csv / figures
scripts/run_phase{1..5}.py, download_real_data.py
notebooks/01_basic_model … 05_el_nino_real_data.ipynb
app/streamlit_app.py, player.py, fire_player.html   # Streamlit app + in-browser fire player
report/report.qmd, user_guide.qmd                   # rendered to HTML
tests/                                              # 63 validation, data and app tests
results/<experiment>/                               # every result saved with its settings and seed
data/processed/                                     # monthly fire counts and Niño 3.4 (small; raw downloads not committed)
```

## Run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                                  # 63 tests, ~16 s
for p in 1 2 3 4; do PYTHONPATH=src python scripts/run_phase$p.py; done   # ~2 min
PYTHONPATH=src python scripts/download_real_data.py     # NASA + NOAA data, ~600 MB, ~4 min (once)
PYTHONPATH=src python scripts/run_phase5.py
streamlit run app/streamlit_app.py
cd report && QUARTO_PYTHON=../.venv/bin/python quarto render report.qmd && quarto render user_guide.qmd
```

Every random draw comes from a fixed seed, so reruns reproduce `results/` exactly (library versions are recorded in
each `settings.json`).

## Deploy your own copy

The app generates its simulations locally and needs no API keys or secrets. Fork this repository, then create a
Streamlit Community Cloud app using branch `main` and entry point `app/streamlit_app.py`; keep Python 3.12. Pushes
to `main` update the app automatically. GitHub Pages publishes the committed report through
`.github/workflows/pages.yml`; in the fork's **Settings → Pages**, select **GitHub Actions** as the source.

**Google Colab:** each notebook's first cell clones `github.com/nhamhhung/forest-fire-spread` if that repository
exists. Otherwise it mounts Google Drive and uses `MyDrive/forest_fire`, so copy the project folder there. Notebook
05 needs `data/processed/`, which is included in the project folder.
