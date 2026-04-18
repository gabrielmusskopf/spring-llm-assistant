import os
import time
import threading
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from configs import *

test_cases = [
    {
        "question": "Como o sistema faz a busca de donos através do sobrenome?",
        "expected_file": "OwnerRepository.java",
        "category": "Data Access"
    },
    {
        "question": "Quais são as regras ou validações ao salvar um novo Pet?",
        "expected_file": "PetValidator.java",
        "category": "Validation"
    },
    {
        "question": "Onde são definidos os tipos de animais (pets) suportados?",
        "expected_file": "PetType.java",
        "category": "Domain Model"
    },
    {
        "question": "Qual endpoint é usado para listar os veterinários do sistema?",
        "expected_file": "VetController.java",
        "category": "Web/API"
    },
    {
        "question": "Como é feito o tratamento de erros e exceções globais?",
        "expected_file": "CrashController.java",
        "category": "Infrastructure"
    },
    {
        "question": "Onde é definida a configuração global de cache para o sistema?",
        "expected_file": "CacheConfiguration.java",
        "category": "Infrastructure"
    },
    {
        "question": "Qual classe lida com o mapeamento de endereços de pessoas (firstName, lastName)?",
        "expected_file": "Person.java",
        "category": "Domain Model"
    },
    {
        "question": "Como é feita a formatação de datas para o nascimento dos pets?",
        "expected_file": "Pet.java",
        "category": "Validation"
    },
    {
        "question": "Quais as permissões e anotações de segurança usadas no VetController?",
        "expected_file": "VetController.java",
        "category": "Security"
    },
    {
        "question": "Onde o sistema define o mapeamento das especialidades dos veterinários?",
        "expected_file": "Specialty.java",
        "category": "Domain Model"
    },
    {
        "question": "Qual o componente responsável por gerenciar a lista de visitas de um animal?",
        "expected_file": "Visit.java",
        "category": "Domain Model"
    },
    {
        "question": "Como o sistema garante que o nome de um tipo de pet não seja vazio?",
        "expected_file": "NamedEntity.java",
        "category": "Validation"
    },
    {
        "question": "Onde estão os métodos para buscar veterinários paginados ou em lista?",
        "expected_file": "VetRepository.java",
        "category": "Data Access"
    },
    {
        "question": "Qual controlador gerencia o fluxo de edição de informações de um dono?",
        "expected_file": "OwnerController.java",
        "category": "Web/API"
    },
    {
        "question": "Como o sistema identifica se uma entidade é nova ou já existe no banco?",
        "expected_file": "BaseEntity.java",
        "category": "Infrastructure"
    }
]

def run_benchmark():
    print(f"Iniciando benchmark de performance")
    print("="*80)
    print(f"Database dir: {DB_DIR}")
    print(f"Benchmark dir: {BENCHMARK_DIR}")
    print(f"Benchmark CSV: {BENCHMARK_CSV}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"MMR K: {MMR_K}")
    print(f"MMR Fetch K: {MMR_FETCH_K}")
    print("="*80)
    
    # Inicialização dos componentes
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'trust_remote_code': True}
    )
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    llm = Ollama(model=LLM_MODEL)

    # Reutilizando o prompt de expansão para garantir o teste real
    expansion_prompt = PromptTemplate(
        input_variables=["question"],
        template=EXPANSION_TEMPLATE
    )

    print(expansion_prompt)

    n_test_cases = len(test_cases)
    i_test = 1
    results_data = []

    for case in test_cases:
        print(f"\n[Testando {i_test}/{n_test_cases}]: {case['question']}")
        
        start_time = time.time()

        # Expansão de Query
        expanded_query = llm.invoke(expansion_prompt.format(question=case['question'])).strip()
        expansion_time = time.time()
        print(expanded_query)
        
        # Busca Vetorial (MMR para diversidade)
        retrieved_docs = vector_db.max_marginal_relevance_search(expanded_query, k=MMR_K, fetch_k=MMR_FETCH_K)
        search_time = time.time()

        # Verificação de Hit (Recuperação)
        retrieved_sources = [doc.metadata.get('source', '') for doc in retrieved_docs]
        # print(retrieved_sources)
        is_hit = any(case['expected_file'] in src for src in retrieved_sources)
        end_time = time.time()
        
        # Latências
        t_exp = expansion_time - start_time
        t_src = search_time - expansion_time
        total_time = end_time - start_time

        results_data.append({
            "Question": case['question'],
            "Category": case['category'],
            "Expected": case['expected_file'],
            "Hit": 1 if is_hit else 0,
            "T_Expansion": round(t_exp, 2),
            "T_Search": round(t_src, 3),
            "Total Time (s)": round(total_time, 2),
            "Expanded Query": expanded_query[:30] + "..."
        })
        i_test = i_test + 1

    # Gerar DataFrame e estatísticas
    df = pd.DataFrame(results_data)
    
    # Cálculo de métricas agregadas
    accuracy = (df["Hit"] == 1).mean() * 100
    avg_latency = df["Total Time (s)"].mean()

    print("\n" + "="*80)
    print("RELATÓRIO DE PERFORMANCE FINAL")
    print("="*80)
    print(df.to_string(index=False))
    print("-" * 80)
    print(f"Precisão (Hit Rate @3): {accuracy:.1f}%")
    print(f"Latência Média Total: {avg_latency:.2f}s")
    print("="*80)

    # Salvar para CSV para criar gráficos no Excel/Python depois
    if not os.path.exists(BENCHMARK_DIR):
        os.mkdir(BENCHMARK_DIR)

    filename = os.path.join(BENCHMARK_DIR, BENCHMARK_CSV)
    df.to_csv(filename, index=False)
    print(f"\nResultados salvos em '{filename}'")


if __name__ == "__main__":
    df = run_benchmark()
