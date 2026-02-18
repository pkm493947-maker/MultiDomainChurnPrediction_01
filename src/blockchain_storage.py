import hashlib
import json
from datetime import datetime

class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []

        # Create the genesis block
        self.create_block(previous_hash="0")

    def create_block(self, previous_hash=None):
        block = {
            "index": len(self.chain) + 1,
            "timestamp": str(datetime.now()),
            "transactions": self.pending_transactions,
            "previous_hash": previous_hash or self.hash(self.chain[-1]) if self.chain else "0"
        }
        self.pending_transactions = []
        self.chain.append(block)
        return block

    def add_transaction(self, transaction):
        """
        Add a new transaction to the list of pending transactions.
        transaction: dict containing info like CustomerID, Risk_Level, Retention_Action, Alert_Message
        """
        self.pending_transactions.append(transaction)
        return self.last_block()["index"] + 1

    def last_block(self):
        return self.chain[-1] if self.chain else None

    @staticmethod
    def hash(block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()
