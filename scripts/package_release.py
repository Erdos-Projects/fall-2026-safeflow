"""Assemble completed research evidence and all three model bundles."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analysis.study_release import config,package_results,package_artifacts
if __name__=='__main__':
    cfg=config();print(package_results(cfg));print(package_artifacts(cfg))
