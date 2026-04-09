# Football Match Outcome Prediction

### Project Overview
This project develops a machine learning model to predict the outcomes of football (soccer) matches. It leverages historical match data, including statistics like goals scored, shots on target, and possession, to train a Random Forest Classifier. The goal is to predict whether a team will win (target = 1) or not (target = 0) in an upcoming match.

### Dataset
The model uses a `matches.csv` dataset, which contains detailed match statistics for various football teams over several seasons. Key features include match date, time, competition, venue, goals scored (gf), goals allowed (ga), shots (sh), shots on target (sot), distance of shots (dist), free kicks (fk), penalties (pk), and penalty attempts (pkatt).

### Methodology
1.  **Data Loading and Initial Exploration**: The `matches.csv` file is loaded into a pandas DataFrame, and initial data types and value counts are inspected.
2.  **Feature Engineering**:
    *   `date` column is converted to datetime objects.
    *   Categorical features `venue` and `opponent` are encoded into numerical representations (`venue` and `opp_code`).
    *   `hour` is extracted from the `time` column.
    *   `day_code` (day of the week) is extracted from the `date` column.
    *   A `target` variable is created: 1 if the `result` is a 'W' (Win), 0 otherwise.
3.  **Rolling Averages**: To capture team form and recent performance, rolling averages (over the last 3 matches, excluding the current match) are calculated for several key statistical features (`gf`, `ga`, `sh`, `sot`, `dist`, `fk`, `pk`, `pkatt`) for each team. These new features are appended to the dataset.
4.  **Model Training**: A Random Forest Classifier is initialized with `n_estimators=50`, `min_samples_split=10`, and `random_state=1`.
5.  **Data Splitting**: The data is split into training and testing sets based on the `date` column: matches before '2022-01-01' are used for training, and matches after '2022-01-01' are used for testing.
6.  **Prediction and Evaluation**: The model is trained on the selected `predictors` (including the new rolling average features) and then used to make predictions on the test set. The model's performance is evaluated using `accuracy_score` and `precision_score`.
7.  **Combined Predictions Analysis**: The predictions are merged with original match details to analyze scenarios where both teams predict a win for themselves and a loss for the opponent.

### Results
*   **Initial Accuracy**: The initial model (without rolling averages) achieved an accuracy of approximately **61.2%**.
*   **Initial Precision**: The initial precision score was approximately **47.5%**.
*   **Precision with Rolling Averages**: After incorporating rolling average features, the precision score improved to **62.5%**.
*   **Analysis of Conflicting Predictions**: When both teams predicted a win for themselves (`predicted_x=1` and `predicted_y=0` - implying the opponent predicted a loss), the `actual_x` (actual outcome for team X) was a win in 12 out of 19 cases, resulting in a precision of approximately **63.2%** for these specific scenarios.

### Usage
To run this project:
1.  Ensure you have the `matches.csv` dataset in the same directory as the notebook.
2.  Execute the notebook cells sequentially.

### Dependencies
*   `pandas`
*   `scikit-learn`
