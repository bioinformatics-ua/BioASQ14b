#!/bin/bash

export PYTHONUNBUFFERED=1 
uv run python pubmed_downloader.py download 2018 2019 2020 2021 2022 2023 2024 -w 12

