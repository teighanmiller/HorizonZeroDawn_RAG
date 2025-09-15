# HorizonZeroDawn_RAG

A Retrieval-Augmented Generation (RAG) system built around _Horizon Zero Dawn_ — combining external knowledge retrieval with generation / language model-based response tailoring.

---

## Table of Contents

- [Project Status](#project-status)
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture & Tech Stack](#architecture--tech-stack)
- [Getting Started](#getting-started)
- [What I Learned & Skills Gained](#what-i-learned--skills-gained)
- [Challenges & Resolutions](#challenges--resolutions)
- [Future Work](#future-work)
- [Why Employers Should Care](#why-employers-should-care)
- [License](#license)

---

## Project Status

**In Development**  
This project is actively being developed and refined. Certain components may not be fully functional yet without proper configuration or access to private keys (e.g., OpenAI Model API's). The repository reflects ongoing progress in building a complete Retrieval-Augmented Generation system.

Employers: This demonstrates my ability to design, iterate, and improve on complex AI/ML projects in real-world scenarios.

---

## Project Overview

The purpose of **HorizonZeroDawn_RAG** is to create a system that:

- Ingests knowledge (e.g. lore, character info, game mechanics) about _Horizon Zero Dawn_.
- Allows user queries (e.g. “Tell me the backstory of Rost,” or “How do machines evolve over time in the game?”).
- Retrieves relevant documents/passages from the stored knowledge.
- Feeds retrieved info into a language model to generate coherent, informed, context-aware answers.

This blends information retrieval + natural language generation — useful for chatbots, game-companion tools, Q&A interfaces, etc.

---

## Features

- **Knowledge Base Construction** – pipeline for collecting, cleaning, and structuring relevant game lore or documentation.
- **Document Retrieval** – indexing with embeddings/vector DBs for similarity search.
- **Prompt / Query Handling** – manages user input and context.
- **Generation Module** – LLM-based responses enriched by retrieved knowledge.
- **Response Refinement** – optional hallucination filtering and summarization.

---

## Architecture & Tech Stack

| Component                                             | Purpose                                             |
| ----------------------------------------------------- | --------------------------------------------------- |
| Data collection & preprocessing                       | Scraping, parsing, and cleaning lore/game documents |
| Embeddings / Vector Store (HuggingFace, Qdrant, BM25) | Hybrid search and retrieval                         |
| LLM module (OpenAI API and open-source LLMs)          | Response generation                                 |
| Backend service                                       | Query handling and serving                          |
| Frontend interface (Streamlit)                        | User interaction                                    |

**Technologies**:

- Python
- `sentence-transformers`, `openai`, vector DBs (Qdrant)
- Streamlit
- Docker / cloud deployment (Will be implemented)

---

## Getting Started

**\*This will not work without adding Azure API credentials for a GPT4o-mini model.**
**\*This is still in development and will be simplified using docker once implemented.**

1. **Clone the repository**

   ```bash
   git clone https://github.com/teighanmiller/HorizonZeroDawn_RAG.git
   cd HorizonZeroDawn_RAG
   ```

2. **Make a .env file**
   The project uses an `.env` file for configuration. Key variables:

   - AZURE_OPENAI_ENDPOINT_URI
   - AZURE_OPENAI_API_KEY
   - AZURE_API_VERSION

3. **Run the Docker Container**

   - Follow the instructions found in README.Docker.md: https://github.com/teighanmiller/HorizonZeroDawn_RAG/blob/main/README.Docker.md

---

## What I Learned & Skills Gained

- **Information Retrieval** — embeddings, vector search, similarity metrics.
- **Prompt Engineering & LLM Usage** — guiding models, avoiding hallucinations, context handling.
- **Data Cleaning / Preparation** — parsing, splitting, and normalizing unstructured text.
- **System Design** — pipeline development: ingestion → indexing → retrieval → generation → serving.
- **Backend Development** — API endpoints, query handling, error management.
- **Iteration & Testing** — refining retrieval and generation quality with evaluation metrics.
- **Problem Solving under Ambiguity** — resolving conflicts in lore data and handling incomplete information.

---

## Challenges & Resolutions

- **Challenge**: Retrieval sometimes mis-ranked irrelevant documents.  
  **Solution**: tested different embeddings, hybrid search.

- **Challenge**: Generated answers risked hallucinations.  
  **Solution**: Constrained prompts to use retrieved context only and added output filtering.

- **Challenge**: Scaling for speed and efficiency.  
  **Solution**: Chunked documents, optimized vector store, cached common queries, cached chat instance.

---

## Future Work

- Add new data sources (developer notes, wikis, interviews).
- Review user feedback for retrieval improvement.
- Explore open-source LLM deployment.
- Implement versioning for updated lore.
- Deploy using AWS.

---
