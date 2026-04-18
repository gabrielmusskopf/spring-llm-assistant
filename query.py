from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from configs import *

def main():
    print("Inicializando RAG com expansão de query")
    print("="*80)
    print(f"Database dir: {DB_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"LLM model: {LLM_MODEL}")
    print(f"MMR K: {MMR_K}")
    print(f"MMR Fetch K: {MMR_FETCH_K}")
    print(f"Show context match: {SHOW_CONTEXT_MATCH}")
    print("="*80)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    llm = Ollama(model=LLM_MODEL)

    expansion_prompt = PromptTemplate(template=EXPANSION_TEMPLATE, input_variables=["question"])
    final_prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

    print("\nTerminal de QA do Código Ativo! (Digite 'sair' para encerrar)\n")

    while True:
        question = input("❯ Sua question: ")
        
        if question.lower() == 'sair':
            break
        if not question.strip():
            continue

         # Expandir a Query
        print("\n  [1/3] Traduzindo a pergunta para contexto de programação...")
        expanded_query = llm.invoke(expansion_prompt.format(question=question)).strip()
        if (SHOW_CONTEXT_MATCH):
            print(f"      ↳ Termos de busca gerados pelo LLM: [{expanded_query}]")

        # Buscar no Vector DB
        print("  [2/3] Buscando vetores no código usando MMR...")
        results = vector_db.max_marginal_relevance_search(expanded_query, k=MMR_K, fetch_k=MMR_FETCH_K)

        texts = []
        for i, doc in enumerate(results):
            texts.append(f"--- Trecho {i+1} ---\n{doc.page_content}\n")

        context = "\n".join(texts)

        # Gerar a Resposta Final
        prompt = final_prompt.format(context=context, question=question)
        if (SHOW_CONTEXT_MATCH):
            print(prompt)

        print("  [3/3] Formulando a explicação final...")
        answer = llm.invoke(prompt)

        print("\n" + "="*60)
        print(answer)
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
