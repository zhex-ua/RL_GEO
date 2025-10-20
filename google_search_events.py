"""
Google Custom Search API scraper for Polymarket events.
Uses multiple API keys with automatic failover when quota is exhausted.
"""

import os
import sys
import json
import time
import argparse
import logging
import csv
import datetime
import pytz
import pandas as pd
import requests
from tqdm import tqdm
from utils import EVENT_META_COLUMNS, TIME_TEMPLATE

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M"

GOOGLE_SEARCH_COLUMNS = [
    "slug", # Event slug
    "searchTerms", # The search query, i.e. response.json['queries']['request'][0]['searchTerms']
    "totalResults", # Total number of results found, i.e. response.json['queries']['request'][0]['totalResults']
    "linkTitle", # Title of the search result, i.e. response.json['items'][i]['title']
    "link", # URL of the search result, i.e. response.json['items'][i]['link']
    "snippet", # Snippet/description of the search result, i.e. response.json['items'][i]['snippet']
    "rank", # Rank of the search result, i.t. startIndex + index in the results
    "timestamp" # Timestamp of when the search was performed
]

logging.basicConfig(
    filename="google_search.log",
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    force=True
)

def log_update(info):
    logging.info(info)
    print(info)


class GoogleSearchScraper:
    """
    Scraper for Google Custom Search API with multiple API key support.
    """

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.metadata_path = os.path.join(self.data_dir, args.metadata_path)
        self.output_path = os.path.join(self.data_dir, args.output_path)

        # Load API keys and search engine ID from environment or args
        self.api_keys = self._load_api_keys(args.api_keys)
        self.search_engine_id = args.search_engine_id or os.getenv('GOOGLE_CSE_ID')

        if not self.search_engine_id:
            raise ValueError("Search engine ID not provided. Set --search_engine_id or GOOGLE_CSE_ID environment variable")

        if not self.api_keys:
            raise ValueError("No API keys provided. Set --api_keys or GOOGLE_API_KEYS environment variable")

        # Track current API key index and usage
        self.current_key_index = 0
        self.key_usage = {key: 0 for key in self.api_keys}

        # Base URL for Google Custom Search API
        self.base_url = "https://www.googleapis.com/customsearch/v1"

        # Maximum results per event (Google API allows max 100 total)
        self.max_results_per_event = args.max_results

        # Load target markets
        self.all_metadata = None
        self.target_mkts = []

        # Track already processed events
        self.processed_slugs = set()

    def _load_api_keys(self, api_keys_arg):
        """Load API keys from argument or environment variable"""
        if api_keys_arg:
            return api_keys_arg if isinstance(api_keys_arg, list) else [k.strip() for k in api_keys_arg.split(',')]

        # Try to load from environment variable
        env_keys = os.getenv('GOOGLE_API_KEYS')
        if env_keys:
            return [k.strip() for k in env_keys.split(',')]

        return []

    def get_current_api_key(self):
        """Get the current API key to use"""
        return self.api_keys[self.current_key_index]

    def rotate_api_key(self):
        """Rotate to the next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        log_update(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")

    def load_mkts_status(self):
        """Load open markets from metadata CSV (similar to sum_sb_single_AI.py)"""
        log_update("Loading markets status...")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.all_metadata = pd.read_csv(self.metadata_path)

        # Replace years that start with 0 (like 0024) with 2024
        if 'end_date' in self.all_metadata.columns:
            self.all_metadata['end_date'] = self.all_metadata['end_date'].str.replace(r'^0024-', '2024-', regex=True)
            self.all_metadata['end_date'] = pd.to_datetime(self.all_metadata['end_date'], utc=True, format='mixed')

        # Get all markets
        target_mkts_df = self.all_metadata
        logging.info(f"Target markets before filtering: {len(target_mkts_df)}")

        # Drop duplicates
        target_mkts_df = target_mkts_df.drop_duplicates(subset=['event_slug'], keep='last')
        logging.info(f"Target markets after dropping duplicates: {len(target_mkts_df)}")

        # Blockchain filter: skip markets that have no event_id
        if 'event_id' in target_mkts_df.columns:
            target_mkts_df = target_mkts_df[target_mkts_df['event_id'].notna()]
            logging.info(f"Target markets after id filter: {len(target_mkts_df)}")

        # Open markets filter (closed == False)
        if 'closed' in target_mkts_df.columns:
            target_mkts_df = target_mkts_df[target_mkts_df['closed'] == False]
            logging.info(f"Target markets after open filter: {len(target_mkts_df)}")

        self.target_mkts = list(zip(target_mkts_df['event_slug'], target_mkts_df['title']))
        log_update(f"Loaded {len(self.target_mkts)} open markets to search")

    def load_processed_slugs(self):
        """Load already processed event slugs from output CSV"""
        if os.path.exists(self.output_path):
            df = pd.read_csv(self.output_path)
            self.processed_slugs = set(df['slug'].values)
            log_update(f"Found {len(self.processed_slugs)} already processed events")
        else:
            # Create new CSV with headers
            with open(self.output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(GOOGLE_SEARCH_COLUMNS)

    def google_search(self, query, max_results=100):
        """
        Perform Google Custom Search API query with pagination.
        Returns up to max_results (max 100 due to API limit).
        """
        all_items = []
        search_terms = query
        total_results = '0'
        num_pages = max_results // 10

        for page in range(num_pages):
            params = {
                'key': self.get_current_api_key(),
                'cx': self.search_engine_id,
                'q': query,
                'num': 10,
                'start': page * 10 + 1
            }

            # Try all API keys if quota exhausted
            for _ in range(len(self.api_keys)):
                try:
                    response = requests.get(self.base_url, params=params, timeout=10)

                    if response.status_code in [429, 403]:
                        self.rotate_api_key()
                        params['key'] = self.get_current_api_key()
                        time.sleep(1)
                        continue

                    response.raise_for_status()
                    self.key_usage[self.get_current_api_key()] += 1

                    data = response.json()
                    if page == 0:
                        req_info = data.get('queries', {}).get('request', [{}])[0]
                        search_terms = req_info.get('searchTerms', query)
                        total_results = req_info.get('totalResults', '0')

                    items = data.get('items', [])
                    all_items.extend(items) # is this order presserving? yes.
                    if len(items) < 10:
                        return {'searchTerms': search_terms, 'totalResults': total_results, 'items': all_items}
                    break

                except Exception as e:
                    logging.error(f"Request error: {e}")
                    time.sleep(2)

            time.sleep(0.5)
        return {'searchTerms': search_terms, 'totalResults': total_results, 'items': all_items}

    def search_event(self, slug, title):
        """Search for a single event and save results."""
        try:
            response = self.google_search(title, max_results=self.max_results_per_event)
            search_terms = response.get('searchTerms', title)
            total_results = response.get('totalResults', '0')
            items = response.get('items', [])
            timestamp = datetime.datetime.now(pytz.utc).strftime(TIME_TEMPLATE)

            with open(self.output_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                if not items:
                    writer.writerow([slug, search_terms, total_results, '', '', '', '', timestamp])
                else:
                    for idx, item in enumerate(items):
                        writer.writerow([
                            slug, search_terms, total_results,
                            item.get('title', ''), item.get('link', ''),
                            item.get('snippet', ''), idx + 1, timestamp
                        ])

            logging.info(f"Successfully searched {slug}: {len(items)} results")
            return True

        except Exception as e:
            logging.error(f"Error searching event {slug}: {e}")
            return False

    def run(self):
        """Main execution loop"""
        log_update("Starting Google Custom Search scraper...")

        # Load markets and processed events
        self.load_mkts_status()
        self.load_processed_slugs()

        # Filter out already processed events
        events_to_search = [(slug, title) for slug, title in self.target_mkts
                           if slug not in self.processed_slugs]

        log_update(f"Events to search: {len(events_to_search)} (skipping {len(self.processed_slugs)} already processed)")

        if len(events_to_search) == 0:
            log_update("No new events to search. Exiting.")
            return

        # Process events with progress bar
        success_count = 0
        error_count = 0

        with tqdm(total=len(events_to_search), desc="Searching events") as pbar:
            for slug, title in events_to_search:
                try:
                    if self.search_event(slug, title):
                        success_count += 1
                    else:
                        error_count += 1

                    pbar.update(1)
                    pbar.set_postfix({
                        'success': success_count,
                        'errors': error_count,
                        'key': f"{self.current_key_index + 1}/{len(self.api_keys)}"
                    })

                except KeyboardInterrupt:
                    log_update("KeyboardInterrupt detected. Exiting...")
                    break
                except Exception as e:
                    logging.error(f"Unexpected error processing {slug}: {e}")
                    error_count += 1
                    pbar.update(1)

        # Print final statistics
        log_update(f"\nCompleted!")
        log_update(f"Success: {success_count}")
        log_update(f"Errors: {error_count}")
        log_update(f"\nAPI Key Usage:")
        for i, key in enumerate(self.api_keys):
            log_update(f"  Key {i + 1}: {self.key_usage[key]} queries")


def main():
    parser = argparse.ArgumentParser(
        description='Scrape Google Search results for Polymarket events using Custom Search API'
    )

    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory where data is stored')
    parser.add_argument('--metadata_path', type=str, default='events_meta.csv',
                       help='Metadata CSV file path (relative to data_dir)')
    parser.add_argument('--output_path', type=str, default='google_search_results.csv',
                       help='Output CSV file path (relative to data_dir)')
    parser.add_argument('--api_keys', type=str, default=\
                        [
                         
                        ],
                       help='Comma-separated list of Google API keys (or set GOOGLE_API_KEYS env var)')
    parser.add_argument('--search_engine_id', type=str, default='5579e146a933d4ed9',
                       help='Google Custom Search Engine ID (or set GOOGLE_CSE_ID env var)')
    parser.add_argument('--max_results', type=int, default=100,
                       help='Maximum results per event (max 100, Google API limit). Each event uses (max_results/10) API calls.')

    args = parser.parse_args()

    try:
        scraper = GoogleSearchScraper(args)
        scraper.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
