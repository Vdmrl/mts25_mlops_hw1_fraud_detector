import os
import pandas as pd
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = CatBoostClassifier()

try:
    model.load_model('./models/my_catboost.cbm')
    logger.info('Pretrained model imported successfully...')
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Define optimal threshold
model_th = 0.98

# Make prediction
def make_pred(dt: pd.DataFrame, path_to_file: str) -> pd.DataFrame:
    if model is None:
        logger.error("Model not loaded, cannot make predictions.")
        return pd.DataFrame()

    # Make submission dataframe
    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': (model.predict_proba(dt)[:, 1] > model_th) * 1
    })
    logger.info('Prediction complete for file: %s', path_to_file)
    return submission

def generate_feature_importances(output_path: str) -> None:
    """Extracts top 5 features and saves them to a JSON file."""
    if model is None:
        logger.error("Model not loaded, cannot generate feature importances.")
        return

    feature_importances = model.get_feature_importance()
    feature_names = model.feature_names_

    importances_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    top_5_features_dict = importances_df.head(5).set_index('feature')['importance'].to_dict()
    features_path = os.path.join(output_path, 'feature_importances.json')

    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump(top_5_features_dict, f, indent=4)
    logger.info(f'Feature importance saved to: {features_path}')

def generate_scores_plot(dt: pd.DataFrame, output_path: str) -> None:
    """Calculates probabilities and saves a plot of their distribution."""
    if model is None:
        logger.error("Model not loaded, cannot generate plot")
        return

    probabilities = model.predict_proba(dt)[:, 1]

    plt.figure(figsize=(10, 6))
    sns.kdeplot(probabilities, fill=True)
    plt.title('Distribution of predicted scores')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plot_path = os.path.join(output_path, 'scores_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f'Scores distribution plot path: {plot_path}')