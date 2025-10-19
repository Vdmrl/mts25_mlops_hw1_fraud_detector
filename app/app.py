import os
import sys
import pandas as pd
import time
import logging
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
# from datetime import datetime

sys.path.append(os.path.abspath('./src'))
from preprocessing import load_train_data, run_preproc
from scorer import make_pred, generate_feature_importances, generate_scores_plot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        self.train = load_train_data()
        logger.info('Service initialized')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            input_df = pd.read_csv(file_path).drop(columns=['name_1', 'name_2', 'street', 'post_code'])

            logger.info('Starting preprocessing')
            processed_df = run_preproc(self.train, input_df)
            
            logger.info('Making prediction')
            submission = make_pred(processed_df, file_path)
            
            logger.info('Prepraring submission file')
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # output_filename = f"predictions_{timestamp}_{os.path.basename(file_path)}"
            output_filename = 'sample_submission.csv'
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info('Predictions saved to: %s', output_filename)

            logger.info('Generating feature importances...')
            generate_feature_importances(self.output_dir)

            logger.info('Generating scores distribution plot...')
            generate_scores_plot(processed_df, self.output_dir)

        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return

# watchdog не работал на винде, пришлось его убрать!
if __name__ == "__main__":
    logger.info('Starting ML scoring service...')

    os.makedirs('/app/input', exist_ok=True)
    os.makedirs('/app/output', exist_ok=True)

    service = ProcessingService()

    processed_files = {}

    logger.info('Starting file polling loop...')
    try:
        while True:
            try:
                all_files = [f for f in os.listdir(service.input_dir) if f.endswith('.csv')]
            except FileNotFoundError:
                logger.warning("Input directory not found. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            for filename in all_files:
                file_path = os.path.join(service.input_dir, filename)

                try:
                    last_modified_time = os.path.getmtime(file_path)
                except FileNotFoundError:
                    continue
                if filename not in processed_files or processed_files[filename] != last_modified_time:
                    logger.info("New or modified file detected: %s", filename)
                    service.process_single_file(file_path)
                    processed_files[filename] = last_modified_time

    except KeyboardInterrupt:
        logger.info('Service stopped by user')