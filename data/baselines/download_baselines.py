from urllib.request import urlopen
from urllib.error import HTTPError
from lxml import etree
from pipemp import StepConverter, BaseProcess, Pipeline, Signals
import json
import urllib.request
import argparse
import gzip
import re
import time
import random

from .pubmed_downloader import parse_article_info


@StepConverter
class MP_PubmedLinks(BaseProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self):
        url = "https://web.archive.org/web/20250222125337/https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
        output = urllib.request.urlopen(url).read().decode("utf-8")
        files = re.findall(r'href="(pubmed\d{2}n\d{4}\.xml\.gz)"', output)

        for file in files:
            print(f"Found: {file}")
            yield url + file


@StepConverter
class MP_PubmedDownloaderAndParser(BaseProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, queue_stream):
        for link in queue_stream:
            print(f"Starting: {link}")
            max_retries = 5

            # FIXED: Added retry logic and sleep to prevent 503 baremetal crashes
            for attempt in range(max_retries):
                try:
                    with gzip.open(urlopen(link)) as xml_fd:
                        for _, element in etree.iterparse(xml_fd, events=("end",)):
                            if element.tag == "MedlineCitation":
                                res = parse_article_info(element)
                                element.clear()
                                if res:
                                    yield {
                                        k: res.get(k, "")
                                        for k in [
                                            "pmid",
                                            "title",
                                            "abstract",
                                            "mesh_terms",
                                            "keywords",
                                        ]
                                    }

                    print(f"Finished: {link}")
                    time.sleep(2)  # Polite sleep to keep NCBI happy
                    break

                except HTTPError as e:
                    if e.code == 503:
                        wait_time = 2 * (random.randrange(1, 100)) / 10 * (2**attempt)
                        print(
                            f"503 Rate Limited on {link}. Sleeping {wait_time}s (Attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                    else:
                        print(f"HTTP Error {e.code} on {link}. Skipping.")
                        break
                except Exception as e:
                    print(
                        f"Connection dropped on {link}: {e}. Sleeping 5s before retry..."
                    )
                    time.sleep(5)
            else:
                print(
                    f"Failed to fetch {link} after {max_retries} attempts. Moving on."
                )


@StepConverter
class MP_JsonlWriter(BaseProcess):
    def __init__(self, jsonl_file, **kwargs):
        super().__init__(**kwargs)
        self.jsonl_file = jsonl_file

    def __call__(self, queue_stream):
        # Using "a" for append mode so it doesn't overwrite if you restart
        with open(self.jsonl_file, "a") as fOut:
            for data in queue_stream:
                fOut.write(f"{json.dumps(data)}\n")
                yield Signals.SAMPLE_CONSUMED


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    pipeline = Pipeline(
        [
            MP_PubmedLinks(num_processes=1),
            MP_PubmedDownloaderAndParser(num_processes=16, size_queue=1000),
            MP_JsonlWriter("ruben_andre_2026.jsonl"),
        ],
        total_samples=40_000_000,
    )

    pipeline.run(debug_inspect_queue_sizes=False)
