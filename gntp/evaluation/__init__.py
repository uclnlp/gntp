# -*- coding: utf-8 -*-

from gntp.evaluation.countries import evaluate_on_countries
from gntp.evaluation.classification import evaluate_classification
from gntp.evaluation.ranking import evaluate
from gntp.evaluation.ranking_show import evaluate_per_predicate
from gntp.evaluation.fast import fast_evaluate

__all__ = [
    'evaluate_on_countries',
    'evaluate',
    'evaluate_classification',
    'evaluate_per_predicate',
    'fast_evaluate'
]
