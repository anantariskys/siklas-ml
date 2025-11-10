import os
import sys

# Tambahkan path folder proyek ke sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

# Import FastAPI app
from main import app as application
