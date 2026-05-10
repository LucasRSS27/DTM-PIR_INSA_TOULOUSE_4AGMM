# DTM-PIR_INSA_TOULOUSE_4AGMM

## Dynamic Topic Modeling (DTM) Analytics

This repository contains a modernized implementation of **Dynamic Topic Modeling**, designed for longitudinal corpus analysis. It features a differentiable inference engine and an interactive dashboard to visualize semantic drift over time.

---

### Key Features

* **Differentiable Inference:** Reinterprets the original Blei & Lafferty (2006) DTM framework using a fully differentiable computation graph powered by **PyTorch**.
* **Kalman Smoothing:** Employs a variational Kalman filter to estimate smooth latent trajectories for both topic-word distributions ($\beta$) and corpus-level topic proportions ($\alpha$).
* **Interactive Dashboard:** A **Shiny for Python** interface for real-time data exploration, filtering, and trend visualization.
* **NLP Pipeline:** Integrated preprocessing using **spaCy**, including lemmatization and automated cleaning.

###  Architecture

1. **`dtm_core.py`**: The engine. Handles the generative process, Kalman forward-backward recursions, and ELBO optimization.
2. **`dash.py`**: The UI. Provides interactive plots for word trends, topic evolution, and model diagnostics.

###  Usage

1. **Install dependencies:** `pip install torch pandas spacy shiny plotly scikit-learn`
2. **Download spaCy model:** `python -m spacy download en_core_web_sm`
3. **Launch the dashboard:**
```bash
shiny run DTM_dash.py

```


4. **Analyze:** Upload your CSV, select your time and text columns, and train the model to discover how your data's themes evolve.

---

*Developed as part of a research project at INSA Toulouse.*
