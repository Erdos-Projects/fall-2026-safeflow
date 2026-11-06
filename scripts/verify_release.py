"""Verify hashes, frozen typical features and all three model bundles."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.analysis.release_verification import verify
if __name__=='__main__':print(json.dumps(verify(),indent=2))
