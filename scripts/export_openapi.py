"""Write the API's OpenAPI schema. The client generates its TypeScript types from it.

Usage: uv run python -m scripts.export_openapi openapi.json
"""

import json
import sys

from app.main import create_app

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "openapi.json"
    with open(path, "w") as fh:
        json.dump(create_app().openapi(), fh, indent=2)
        fh.write("\n")
    print(f"wrote {path}")
