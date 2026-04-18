import os

# Ingestion
DATA_DIR = os.getenv("DATA_DIR", "./spring-petclinic")
DB_DIR = os.getenv("DB_DIR", "./db")
INGESTION_IGNORE_PATTERN = os.getenv("INGESTION_IGNORE_PATTERN", "").split(",")

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "microsoft/unixcoder-base")

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

CHUNK_SIZE = os.getenv("CHUNK_SIZE", 1200)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 200)
MMR_K = os.getenv("MMR_K", 5)
MMR_FETCH_K = os.getenv("MMR_FETCH_K", 15)

# Query
SHOW_CONTEXT_MATCH = os.getenv("SHOW_CONTEXT_MATCH", "True") in ("True", "true", "1", "yes", "y")

# Templates
EXPANSION_TEMPLATE = """Você é um tradutor técnico focado no framework Spring Boot.
Transforme a question do usuário em uma lista curta de termos técnicos de busca em inglês.

REGRAS ABSOLUTAS:
- NUNCA escreva blocos de código.
- NUNCA escreva assinaturas de métodos completas.
- NUNCA use pontuação.
- Retorne no máximo 8 palavras soltas separadas por espaço.

Você deve identificar a Entidade, a Ação e a Camada, todos em inglês.
Se a pergunta envolver persistência, use o sufixo Repository.
Se envolver web, use Controller.
Se envolver validação, use Validator
Converta termos em português para camelCase em inglês.

--- AGORA É A SUA VEZ ---
Pergunta: {question}
Busca:"""

TEMPLATE = """Você é um especialista em Java.
Use APENAS os trechos de código abaixo para responder à question do desenvolvedor.
Se a answer não estiver nos trechos, diga: "Não encontrei essa informação no código."
Seja direto, cite o nome dos arquivos e como o código funciona.
Busque passar uma visão do todo o fluxo necessário.

TRECHOS DE CÓDIGO RECUPERADOS:
{context}

PERGUNTA ORIGINAL: {question}

FORMATO DE RESPOSTA: Plain text para ser exibido no terminal

SUA RESPOSTA TÉCNICA:"""

# Benchmark
BENCHMARK_CSV = os.getenv("BENCHMARK_CSV", "benchmark_results.csv")
BENCHMARK_DIR = os.getenv("BENCHMARK_DIR", "./benchmark")

# Dashboard
CHARTS_DIR = os.getenv("CHARTS_DIR", "./charts")
