import re

import requests

from typing import List, Dict

from logger_config import logger


def search_by_http(query: str, host: str = 'localhost', port: int = 8090) -> List[Dict]:
    url = f"http://{host}:{port}"
    response = requests.post(url, json={'query': query})

    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to get a response. Status code: {response.status_code}")
        return []

def _normalize_subquery(subquery: str) -> str:
    subquery = subquery.strip()
    if subquery.startswith('"') and subquery.endswith('"'):
        subquery = subquery[1:-1]
    if subquery.startswith('Intermediate query'):
        subquery = re.sub(r'^Intermediate query \d+: ', '', subquery)

    return subquery