# MailGuard — Email Spam Detection Using Machine Learning

A Streamlit dashboard that classifies an email's text as **Spam** or **Not Spam (Ham)** using a Scikit-learn pipeline: TF-IDF text vectorization + Multinomial Naive Bayes.

## Project files

- `emails.csv` — 50 illustrative, labelled email examples (25 spam / 25 ham).
- `train_model.py` — validates the CSV, creates a stratified train/test split, trains the pipeline, calculates evaluation metrics, and saves the model and metrics.
- `app.py` — Streamlit dashboard with Futuristic dark-themed Overview, Live Scanner, Model Analytics, Test History, and About pages.
- `requirements.txt` — Python dependencies.
- `Email_Spam_Detection_Presentation.pptx` — school presentation.

## Run on Windows (VS Code)

1. Install Python 3.10 or newer.
2. Extract the ZIP and open the project folder in VS Code.
3. Open the VS Code terminal and run:

   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

4. Train the model and create the evaluation files:

   ```bash
   python train_model.py
   ```

5. Launch the dashboard:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Model workflow

1. Read and validate labelled email text from `emails.csv`.
2. Split the examples into training and test sets (75% / 25%, stratified).
3. Use TF-IDF with unigrams and bigrams to turn text into numeric features.
4. Train Multinomial Naive Bayes on the training examples.
5. Evaluate predictions on the held-out test split using accuracy, spam precision, spam recall, spam F1-score, a confusion matrix, and a classification report.
6. Save the complete Scikit-learn pipeline to `spam_model.joblib` and evaluation details to `model_metrics.json`.

## Dashboard features

- Email classification with a confidence estimate and class probabilities.
- Test-set performance metrics and confusion matrix.
- Session-only test history with CSV export.
- Downloadable evaluation metrics.

## Important limitations

The supplied 50-message dataset is a small, illustrative classroom sample. It is not representative of real email traffic, and its test metrics should not be presented as real-world performance. A model confidence estimate is not a safety guarantee. Do not open suspicious links or attachments, or share passwords or OTPs based on this tool's prediction.

For a stronger research project, use a larger, properly licensed public dataset, check for duplicates and class imbalance, apply cross-validation, compare multiple algorithms, and evaluate on a separate dataset that was not used during development.
