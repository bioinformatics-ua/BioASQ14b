from collections import defaultdict
from typing import Literal

type Year = Literal[
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
    "2026",
]

type PMID = str
type IDsPerBaseline = defaultdict[Year, dict[PMID, int]]
