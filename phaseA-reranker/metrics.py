from ranker_trainer import RankingEvalPrediction
from collections import defaultdict
from ranx import Qrels, Run
from ranx import evaluate


class RanxMetrics:
    def __init__(self, qrels_dict):
        self.qrels_dict = qrels_dict
        # ids_per_batch

    def __call__(self, evaluationOutput: RankingEvalPrediction):

        run_dict = defaultdict(dict)
        for i, metadata in enumerate(evaluationOutput.ranking_metadata):
            run_dict[metadata["id"]][metadata["doc_id"]] = evaluationOutput.predictions[
                i
            ]

        # qrels = Qrels({k:v for k,v in self.qrels_dict.items() if k in run_dict})

        qrels = Qrels(self.qrels_dict)
        run = Run(run_dict)

        return evaluate(
            qrels,
            run,
            [
                "ndcg@5",
                "mrr",
                "recall@10",
                "recall@100",
                "recall@1000",
                "map@10",
                "map-bioasq@10",
            ],
        )
