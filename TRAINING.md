# SAGE: Reproducibility Guide

This guide covers the entire SAGE pipeline in exact chronological order, from environment setup to the final statistical tables. 

All steps run locally except those explicitly mentioned with Kaggle GPU.

---

### 1. Environment Setup (Local)
This step clones the SAGE repository and installs all required Python dependencies. It ensures your local environment has the exact package versions needed to run the interpretability and adversarial robustness scripts.

* **Run:** 
  1. `git clone https://github.com/sakshi-kadian/sage.git`
  2. `cd sage`
  3. `pip install -r requirements.txt`

---

### 2. Data Preparation (Local)
This step downloads the raw Anthropic HH-RLHF and Stanford Human Preferences (SHP) datasets from HuggingFace. It then extracts a carefully stratified 10,000-example subset of HH-RLHF for rapid training, leaving 1,000 examples as an In-Distribution test set, while holding out SHP as the unseen Out-of-Distribution test set.

* **Run:** 
  1. `python data/download_hh_rlhf.py`
  2. `python data/stratified_sampler.py`
  3. `python data/download_shp.py`
* **Output:** 
  - `data/processed/train_10k.parquet`
  - `data/processed/test_1k.parquet`
  - `data/shp_test_subset.json`

---

### 3. Sanity Check - Local Overfit Test (Local)
Before committing to hours of cloud GPU compute, this script verifies that the PyTorch setup is working. It overfits the DeBERTa reward model on a single batch of data, confirming that the forward pass, backward pass, and Bradley-Terry ranking loss functions are all operating correctly.

* **Run:** 
  1. `python scripts/sanity_check.py`
* **Output:** 
  - Terminal logs (loss should drop from ~0.693 to near zero)

---

### 4. Baseline Reward Model Training [Kaggle GPU]
This is the main cloud training step for the baseline model. It leverages Kaggle's dual T4 GPUs to fine-tune the 434-million parameter `microsoft/deberta-v3-large` model on the 10,000-example HH-RLHF dataset for 3 epochs.

* **Run:** 
  1. `notebooks/kaggle_training_setup.ipynb` *([See Kaggle Reference Guide below](#kaggle-reference-guide))*
* **Output:** 
  - `checkpoints/baseline_epoch_3.pt`

---

### 5. Evaluate Baseline Accuracy (Local)
This script evaluates the trained baseline model against the 1,000 held-out HH-RLHF test examples. It computes the exact 59.80% test accuracy and generates plots showing how the model distributes reward scores across chosen vs. rejected text.

* **Run:** 
  1. `notebooks/01_baseline_evaluation.ipynb`
* **Output:**
  - `results/baseline_metrics.json`
  - `results/baseline_test_results.csv`
  - `results/01(1)_figure_reward_distribution.png`
  - `results/01(2)_figure_reward_gap.png`

---

### 6. Mechanistic Interpretability: Attention Rollout (Local)
This step breaks open the "black box" of the reward model. It applies residual-aware Attention Rollout across all 24 DeBERTa layers to trace exactly which specific words and tokens the model heavily indexes on to assign its final reward score.

* **Run:** 
  1. `notebooks/02_attention_rollout.ipynb`
* **Output:**
  - `results/02(1)_figure_chosen_influence.png`
  - `results/02(2)_figure_rejected_influence.png`
  - `results/02(3)_figure_1_attention_heatmaps.png` (**Figure 1**)
  - `results/02(4)_figure_rollout_heatmap.png`

---

### 7. Adversarial Attacks & Dual Stealth Filter (Local)

#### Step 7.1: Exploratory Attack Development
This phase covers the development and prototyping of the three primary attacks: the TextFooler-style Black-Box attack, the Attention Rollout-guided Mechanistic attack, and the strict dual SBERT/GPT-2 stealth filters that ensure the attacks remain undetectable.

* **Run:** 
  1. `notebooks/03_blackbox_attack.ipynb`
  2. `notebooks/04_mechanistic_attack.ipynb`
  3. `notebooks/05_stealth_filters.ipynb`

#### Step 7.2: Large-Scale Attack Evaluation
This executes the final, context-aware BERT Masked Language Model (MLM) attack alongside the other attacks across 100 samples. It checks every generated adversarial text against the dual stealth filter to calculate the final 52% success rate. **Note:** `evaluate_attacks.py` explicitly looks for `models/multi_seed/baseline_seed_1.pt`. If you are running the steps in chronological order, you will not have this checkpoint yet. Please place your `checkpoints/baseline_epoch_3.pt` at that path first, or simply re-run this step after completing Step 14.

* **Run:** 
  1. `python src/adversarial/evaluate_attacks.py`
* **Output:** 
  - `results/attack_eval_results.csv`

#### Step 7.3: Generate Attack Comparison Chart
This script takes the results from the large-scale evaluation and generates a bar chart comparing the stealth pass rates and average reward drops across the different attack methodologies.

* **Run:** 
  1. `notebooks/06_attack_results.ipynb`
* **Output:** 
  - `results/06(1)_figure_2_attack_comparison.png` (**Figure 2**)

---

### 8. Generate the Adversarial Training Dataset (Local)
To harden the model against the vulnerabilities discovered in Step 7, this script merges the clean training data with verified, stealth-filtered adversarial text pairs. This creates the "poisoned" dataset that will be used for defense ablation.

* **Run:** 
  1. `notebooks/07_adversarial_dataset.ipynb`
* **Output:**
  - `data/adv_training_pairs.json`
  - `results/07(1)_dataset_composition.png`

---

### 9. Defense Ablation Training [Kaggle GPU]
This cloud training step conducts the primary ablation study. It fine-tunes the baseline model on 4 different adversarial mixing ratios (10%, 25%, 50%, 100%) to determine the optimal balance between adversarial robustness and clean data accuracy.

* **Run:** 
  1. `notebooks/08_kaggle_defense_ablation.ipynb` *([See Kaggle Reference Guide below](#kaggle-reference-guide))*
* **Output:**
  - Saved checkpoints in `models/ablation/`
  - `results/ablation_training_summary.json`

---

### 10. Evaluate Defense Ablation (Local)
This step evaluates the ablation checkpoints to analyze the precise trade-off curve between attack success rate and the model's accuracy on clean data. It identifies the 25% ratio as the mathematical sweet spot.

* **Run:** 
  1. `notebooks/09_ablation_results.ipynb`
* **Output:**
  - `results/09(1)_figure_3_ablation_tradeoff.png`
  - `results/ablation_tradeoff.csv`

---

### 11. Evaluate Out-of-Distribution Transfer (Local)
A critical test to ensure the adversarial training did not cause catastrophic forgetting. This script compares the baseline model and the 25% defended model against the Stanford SHP dataset—a distribution completely unseen during training.

* **Run:** 
  1. `notebooks/11_ood_transfer.ipynb`
* **Output:**
  - `results/11(1)_figure_4_ood_transfer.png`
  - `results/ood_results.json`

---

### 12. Evaluate Defended Model on Test Set (Local)
This script runs a full inference pass of the 25% defended model on the standard 1,000-example HH-RLHF test set. The resulting predictions are required as input for the subsequent statistical significance tests.

* **Run:** 
  1. `python src/robustness/evaluate_defended.py`
* **Output:** 
  - `results/defended_test_results.csv`

---

### 13. Statistical Significance Tests (Local)
This notebook rigorously verifies that the defensive improvements are not a random fluke. It runs McNemar's Chi-Squared test comparing the Baseline vs. 25% Defended model and computes 95% Bootstrap Confidence Intervals for the paper.

* **Run:** 
  1. `notebooks/10_statistics.ipynb`
* **Output:**
  - `results/10(1)_table_1_final_results.csv`
  - `results/10(2)_table_2_ablation_results.csv`

---

### 14. Multi-Seed Validation Training [Kaggle GPU]
To ensure extreme scientific rigor, this step re-trains both the Baseline (0%) and Best Defense (25%) models across 3 completely random initial seeds.

* **Run:** 
  1. `notebooks/12_kaggle_multiseed_training.ipynb` *([See Kaggle Reference Guide below](#kaggle-reference-guide))*
* **Output:**
  - 6 saved checkpoints in `models/multi_seed/`
  - `results/multi_seed_training_summary.json`

---

### 15. Evaluate All Models (Clean vs. Robustness) (Local)
This massive evaluation script runs all 6 multi-seed checkpoints against the Anthropic HH-RLHF test set to verify their clean In-Distribution accuracy, and evaluates their robustness against the MLM adversarial attack. **Note:** Due to a naming quirk, the internal code prints "OOD Accuracy" during execution, but it is actually calculating the HH-RLHF In-Distribution accuracy. Stanford SHP OOD transfer is handled exclusively in Step 11. 

* **Run:** 
  1. `python src/robustness/evaluate_multiseed.py`
* **Output:**
  - `results/multi_seed_results.json`
  - `results/multi_seed_results.csv`

---

### 16. Final Multi-Seed Figures and Table (Local)
This final notebook parses the multi-seed JSON data, calculates the `mean ± std` across all 3 seeds, and plots the final publication-quality figures that appear in the SAGE paper.

* **Run:** 
  1. `notebooks/13_multiseed_evaluation.ipynb`
* **Output:** 
  - `results/13(1)_table_1_multiseed_results.csv` (**Table 1**)
  - `results/13(2)_figure_3_ablation_with_errorbars.png` (**Figure 3**)
  - `results/13(3)_figure_4_ood_with_errorbars.png` (**Figure 4**)

---

### Kaggle Reference Guide

#### How to Prepare Kaggle Datasets

* **Note for Step 4:** The `kaggle_training_setup.ipynb` notebook clones your repository directly from GitHub and downloads the raw data automatically. You do **not** need to manually upload datasets for this step.
* **Note for Steps 9 and 14:** For the advanced defense training, you must manually upload your local files to Kaggle to use as datasets:
1. Upload your `src/` folder to Kaggle Datasets -> name it `sage-src`.
2. Upload your data folder (`data/`) to Kaggle Datasets -> name it `sage-data`. 
   *(Note: This folder contains the `adv_training_pairs.json` generated in Step 8, which is required).*
3. Upload your trained baseline model (`checkpoints/baseline_epoch_3.pt`) to Kaggle Datasets -> name it `sage-model`.
4. In your Kaggle notebook, click **Add Data** and attach these 3 datasets.

#### How to Run Notebooks
To execute your training code in the Kaggle editor:
1. Go to **Settings** (top menu) -> **Accelerator** -> select **GPU T4 x2**.
2. Click the **Run All** button (double-play icon) at the top of the editor, or run each cell individually.
*(Note: Keep your browser tab open while it runs, or the session may disconnect.)*

#### How to Download Outputs
Only do this **AFTER** your training has finished running in the editor. Do not mix this up with running the code!
1. Click **Save Version** in the top-right corner of the Kaggle notebook.
2. In the dialog box, select **Quick Save** as the Version Type.
3. Click on **Advanced Settings** and select **Save output for this version when creating a Quick Save**.
4. Click the blue **Save** button.
5. Once saved, click the menu in the top-left, go to **Your Work**, and click on your notebook.
6. Navigate to the **Output** tab in the notebook view to see all generated files.
7. Click the **Download** icon next to the files you need (or **Download All** for a zip).

---

*All figures and tables generated by completing the steps above are available in the `results/` directory.*
