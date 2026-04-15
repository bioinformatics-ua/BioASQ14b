
# Type aliases
"""
# Example for PairWise :
{
        "query_001": {
            "question": "What is the treatment for diabetes?",
            0: [  # Negative docs (BM25 retrieved, not relevant)
                {"id": "doc_001", "score": 25.5, "text": "Diabetes is a chronic condition that affects millions worldwide. Risk factors include genetics and obesity."},
                {"id": "doc_002", "score": 24.1, "text": "Metformin is commonly prescribed for type 2 diabetes management."},
                {"id": "doc_003", "score": 22.8, "text": "Regular exercise helps prevent diabetes complications."},
            ],
            1: [  # Positive docs (actually relevant/gold standard)
                {"id": "doc_004", "text": "Insulin therapy remains the primary treatment for type 1 diabetes, with dosages adjusted based on blood glucose monitoring."},
            ]
        },
        "query_002": {
            "question": "What are the symptoms of COVID-19?",
            0: [  # Negative docs
                {"id": "doc_005", "score": 28.2, "text": "The common cold and flu share many symptoms with respiratory viruses."},
                {"id": "doc_006", "score": 26.7, "text": "Vaccination programs have reduced the severity of COVID-19 cases worldwide."},
                {"id": "doc_007", "score": 25.3, "text": "Public health measures were implemented during the pandemic."},
            ],
            1: [  # Positive docs
                {"id": "doc_008", "text": "COVID-19 symptoms include fever, cough, fatigue, loss of taste or smell, and difficulty breathing."},
                {"id": "doc_009", "text": "Severe cases may present with pneumonia, acute respiratory distress syndrome, and organ failure."},
            ]
        }
    }
"""
type SliceDataset = dict[str, dict[str | int, list[dict[str, str]] | str]]

"""
# Example:
{
    "12345": "Diabetes is a chronic disease that affects how your body...",
    "67890": "Type 2 diabetes management includes lifestyle changes...",
    "11111": "Insulin therapy is the primary treatment for type 1 diabetes...",
}
"""
type Collection = dict[str, str]

"""
# Example:
{
    "5c0aab9a7c78d2d783f84346": {  # query ID
        "11111": 1,  # doc 11111 is relevant (label 1)
        "12345": 0,  # doc 12345 is not relevant (label 0)
        "67890": 0,
    },
    "5c0aab9b7c78d2d783f84347": {
        "33333": 1,
        "22222": 0,
    },
}
"""
type QrelsDict = dict[str, dict[str, int]]


"""
# Training example (pointwise):
{
    "id": "5c0aab9a7c78d2d783f84346",
    "query_text": "What is the treatment for diabetes?",
    "doc_text": "Insulin therapy is the primary treatment...",
    "label": 1,  # 1 = relevant, 0 = not relevant
}
# Inference example:
{
    "id": "5c0aab9a7c78d2d783f84346",
    "doc_id": "11111",
    "query_text": "What is the treatment for diabetes?",
    "doc_text": "Insulin therapy is the primary treatment...",
}
"""
type Sample = dict[str, str | int]
"""
# Pointwise training (after tokenization):
{
    "input_ids": [101, 2054, 2003, 1996, 4524, ...],  # token IDs
    "attention_mask": [1, 1, 1, 1, 1, ...],
    "token_type_ids": [0, 0, 0, 0, 0, ...],
    "labels": 1,  # relevance label
}
# Pairwise training (after tokenization):
{
    "pos_inputs": {  # positive document sample
        "input_ids": [101, 2054, 2003, ...],
        "attention_mask": [1, 1, 1, ...],
    },
    "neg_inputs": {  # negative document sample
        "input_ids": [101, 2054, 2003, ...],
        "attention_mask": [1, 1, 1, ...],
    },
}
# Multi-negative pairwise training:
{
    "pos_inputs": {  # single positive
        "input_ids": [101, 2054, ...],
        "attention_mask": [1, 1, ...],
    },
    "neg_inputs": [  # list of multiple negatives
        {"input_ids": [101, 2054, ...], "attention_mask": [1, 1, ...]},
        {"input_ids": [101, 2054, ...], "attention_mask": [1, 1, ...]},
        {"input_ids": [101, 2054, ...], "attention_mask": [1, 1, ...]},
    ],
}
"""
type ProcessedSample = dict[str, Sample | list[Sample] | str | int]
