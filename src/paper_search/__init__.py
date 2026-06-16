"""TopConf Paper Search.

A small, conference-focused paper search MVP:
- parse saved conference HTML into a normalized CSV
- build FAISS indices over title / abstract / both
- search with keyword, semantic, or hybrid retrieval
"""

__version__ = "0.1.0"
