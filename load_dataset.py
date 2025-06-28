from datasets import Dataset, load_dataset

from logger_config import logger

# from logger_config import logger

logger.info(f'Installing Dataset HotPotQA from corag/multihopqa...')

ds: Dataset = load_dataset("corag/multihopqa", "hotpotqa", split="validation")

logger.info(f'Dataset successfully loaded!')
logger.info(ds)

logger.info('Remove the subqueries, subanswers and corag answer predictions from the database')
ds = ds.remove_columns([name for name in ['subqueries', 'subanswers', 'predictions'] if name in ds.column_names])
logger.info('Add a column with a task description(with default value)')
ds = ds.add_column('task_desc', ['answer multi-hop questions' for _ in range(len(ds))])

# logger.info()

global total_cnt
total_cnt = len(ds)

ds

ds = ds.select(range(5))
ds[1]