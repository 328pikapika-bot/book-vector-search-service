# Book Vector Search Service

English | [简体中文](README.md)

Structures, cleans, chunks, and embeds book chapters, then serves content search and maintenance APIs backed by Milvus and MySQL.

## Capabilities

- Ingests, deletes, and searches books and chapters while retaining source fields and filters.
- Includes same-book operation locks, concurrency controls, capacity switching, and failure compensation.
- Publishes only the selected standalone service, excluding earlier graph and web projects.

## Code and setup

The only included service tree is `源码/05_编码/skwx_ai-python/social_sciences_vector/`. `main_library_api.py` is the API entry, `national_development/` contains book operations, and `scripts/` contains preflight and validation tools. Dependencies are in that service's `requirements.txt`.

Provide Python, Milvus, MySQL, an embedding model, and a model path appropriate to your machine. Review `config.py`, `constant.py`, and startup scripts before running. The latest explicit release in the source material is v2.2.0; this repository uses the current standalone code tree rather than assembling historical release archives.

## Configuration

MySQL uses `BOOK_VECTOR_MYSQL_HOST`, `BOOK_VECTOR_MYSQL_USER`, `BOOK_VECTOR_MYSQL_PASSWORD`, and `BOOK_VECTOR_MYSQL_DATABASE`. Supply Milvus, Neo4j, and ZhipuAI settings from `.env.example` through the process environment. The code does not load `.env` automatically. Example addresses are not production endpoints.

## Verification and limits

Machine-specific model paths must be configured. Real model loading, MySQL/Milvus connectivity, concurrency, and search data were not verified during the 2026-11-04 public-code review. Historical logs, old projects, and release archives are excluded.

See the [publication notes](仓库发布说明.md) for the included files.
