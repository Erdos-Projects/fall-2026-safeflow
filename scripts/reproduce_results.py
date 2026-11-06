"""Full raw-data replay of the recorded models, calibration and test protocol."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.analysis.release_pipeline import replay,validate
from src.analysis.study_release import config
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--use-cache',action='store_true');p.add_argument('--check',action='store_true')
    a=p.parse_args()
    if a.check:validate(config());print('Frozen replay configuration valid; no data opened or models fitted.')
    else:print(replay(a.prepare_only,a.use_cache))
