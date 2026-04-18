import os
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from configs import *

def main():
    print("Iniciando pipeline de ingestão")
    print("="*80)
    print(f"Data dir: {DATA_DIR}")
    print(f"Database dir: {DB_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Ignorar arquivos extra: {INGESTION_IGNORE_PATTERN}")
    print("="*80)
          
    # Carregando todos os arquivos .java
    loader = DirectoryLoader(
        DATA_DIR, 
        glob="**/*.java", 
        loader_cls=TextLoader,
        show_progress=True
    )
    raw_docs = loader.load()

    # Filtro arquitetural e limpeza
    clean_docs = []
    print("\nAplicando filtros de arquitetura e limpando boilerplate...")
    
    for doc in raw_docs:
        # Padroniza as barras do caminho do arquivo (Windows/Linux)
        path = doc.metadata.get('source', '').replace("\\", "/")
        
        # Ignorar diretório e arquivos de testes e package-info
        if "/test/" in path or "Test.java" in path or "Tests.java" in path or "package-info.java" in path or path in INGESTION_IGNORE_PATTERN:
            continue

        text = doc.page_content
        
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)          # Limpeza de licenças (blocos /* ... */ no topo)
        text = re.sub(r'^import .*?;', '', text, flags=re.MULTILINE)    # Limpeza de imports
        text = re.sub(r'\n\s*\n', '\n\n', text)                         # Limpeza de quebra de linhas
        
        # Injeção de Contexto
        # Pega apenas o nome final do arquivo (ex: Owner.java)
        class_name = os.path.basename(path)
        # Força o nome da classe na primeira linha do chunk para guiar o modelo matemático
        rich_text = f"// Arquivo-Origem: {class_name}\n" + text.strip()

        # Atualiza o documento e adiciona na lista válida
        doc.page_content = rich_text
        clean_docs.append(doc)

    print(f"De {len(raw_docs)} arquivos totais, {len(clean_docs)} são código de produção válidos.")

    # Quebra em Chunks (Syntax-aware para Java)
    print("\nQuebrando código em chunks considerando a sintaxe Java...")
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(clean_docs)
    print(f"Código de produção dividido em {len(chunks)} chunks semânticos.")

    # Gerando vetores e salvando no ChromaDB
    print("\nGerando vetores e persistindo no ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'trust_remote_code': True})
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print("\nIngestão otimizada concluída com sucesso! Banco atualizado.")

if __name__ == "__main__":
    main()
