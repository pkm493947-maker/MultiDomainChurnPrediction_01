import json
import os
import hashlib
from datetime import datetime


BLOCKCHAIN_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../blockchain/ledger.json"
)


def calculate_hash(data):
    """Generate SHA256 hash for blockchain integrity"""
    data_string = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_string.encode()).hexdigest()


def store_blockchain_record(data):

    print("🔗 Creating Blockchain Record...")

    # Create blockchain folder if not exist
    os.makedirs(os.path.dirname(BLOCKCHAIN_FILE), exist_ok=True)

    # Load existing ledger
    if os.path.exists(BLOCKCHAIN_FILE):
        with open(BLOCKCHAIN_FILE, "r") as f:
            ledger = json.load(f)
    else:
        ledger = []

    # Create record
    record = {
        "timestamp": str(datetime.now()),
        "data": data,
        "hash": calculate_hash(data)
    }

    ledger.append(record)

    # Save ledger
    with open(BLOCKCHAIN_FILE, "w") as f:
        json.dump(ledger, f, indent=4)

    print("✅ Blockchain Record Saved")
    print("📂 Location:", BLOCKCHAIN_FILE)