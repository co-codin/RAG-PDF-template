"""rag_app: a production-style Retrieval-Augmented Generation service.

Layering (dependencies point downwards only):

    api            HTTP: routers, schemas, DI, middleware
    services       use-cases (ingest, query, documents) - framework-agnostic
    rag            RAG logic: loaders, chunking, retrieval, grading, prompts, graph
    infrastructure swappable backends behind Protocols (embeddings, llm, vector store,
                   checkpointer, job store) + env-driven factories
    domain / core  shared models & pure helpers / config, logging, security, errors
"""

__version__ = "0.2.0"
