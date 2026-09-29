import os
import sys

# Make the repository root importable (models/, jlens_motion/) under plain `pytest`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
