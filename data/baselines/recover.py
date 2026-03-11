import gzip
import json
from urllib.request import urlopen
from lxml import etree

# Safely import your existing parsing logic
from download_baselines import parse_article_info


def recover_single_file():
    link = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed26n0652.xml.gz"
    out_file = "pubmed_baseline_2026.jsonl"

    print(f"Attempting to recover: {link}")

    # Open in APPEND mode ("a") to safely add to your existing data
    with open(out_file, "a") as fOut, gzip.open(urlopen(link)) as xml_fd:
        for _, element in etree.iterparse(xml_fd, events=("end",)):
            if element.tag == "MedlineCitation":
                res = parse_article_info(element)
                element.clear()
                if res:
                    # Filter to only the columns you want
                    filtered_res = {
                        k: res.get(k, "")
                        for k in ["pmid", "title", "abstract", "mesh_terms", "keywords"]
                    }
                    fOut.write(json.dumps(filtered_res) + "\n")

    print(f"Success! {link} has been appended to {out_file}.")


if __name__ == "__main__":
    recover_single_file()
