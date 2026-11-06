"""Retain existing analysis test entry points without duplicating their code."""
import unittest

def load_tests(loader,tests,pattern):
    for name in ['src.analysis.test_early_warning_errors','src.analysis.test_paired_uncertainty','src.analysis.test_evaluate_holdout','src.analysis.test_local_benchmark']:
        tests.addTests(loader.loadTestsFromName(name))
    return tests
