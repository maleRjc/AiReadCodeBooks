#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AiReadCodeBooks - Catalog Updater
Syncs README tables and statistics from scripts/book_matrix_100.json.
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_FILE = os.path.join(BASE_DIR, "scripts", "book_matrix_100.json")

def main():
    with open(MATRIX_FILE, "r", encoding="utf-8") as f:
        matrix = json.load(f)
    print(f"Loaded {len(matrix)} books from matrix.")
    available = [b for b in matrix if b.get("status") == "available"]
    queued = [b for b in matrix if b.get("status") != "available"]
    print(f"Status: {len(available)} available, {len(queued)} queued.")

if __name__ == "__main__":
    main()
