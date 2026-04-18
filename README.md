# Desafio Laboratório de IA

- [Visão geral](#visão-geral)
- [Criação e evolução](#criação-e-evolução)
- [Benchmark](#benchmark)
- [Demonstração de uso](#demonstração-de-uso)
  - [Parâmetros](#parâmetros)
  - [Screenshots](#screenshots)
- [Passos para usar em produção](#passos-para-usar-em-produção)

A egenharia de software atualmente é, via de regra, muito complexa, envolvendo diversos sistemas, padrões distintos e regras de negócio espalhadas por microserviços distintos. Regras de negócio essas que muitas vezes ficam mescladas ao código "téncico", fragmentando o negócio em camadas de abstração. Junto a isso, a documentação tradicional se torna cada vez mais obsoleta, geralmente não acompanhando a velocidade de alteração de um software. Tudo isso contribui para uma carga cognitiva maior durante o onboarding de um novo engenheiro, ilhas de conhecimento, onde partes do código, principalmente mais antigas, sejam de conhecimento de poucos.

Então, o código fonte deve ser a fonte da verdade, ele também é um artefato de documentação do sistema. Mas em sistemas grandes e com muitos desenvolvedores isso se torna impossível de ser feito. Não há tempo hábil para consultar o código fonte toda vez que precisar de uma resposta rápida sobre uma regra de negócio. 

Esse desafio tem como objetivo criar um sistema de perguntas e respostas sobre um código fonte, utilziando de tecnologias como modelos de linguagem, banco vetorial, e técnicas de RAG, de forma local e privada. Também, será documentado o processo evolutivo da criação desse sistema, apresentando as limitações de cada abordagem e a análise do resultado final. Para aumentar o foco, será consideradod somente sistemas Java que utilizam Spring Framework, que é uma tecnologia muito popular no mercado de trabalho. Para os testes, foi usado o repositório de código aberto do Sprint PetClinic.

## Visão geral

![Ingestão](./docs/ingestao.png)

![Consulta](./docs/consulta.png)

## Criação e evolução

O desenvolvimento deste assistente de código não seguiu um caminho linear. A primeira abordagem do projeto foi utilizar Naive RAG (RAG Ingênuo), empregando o modelo de embedding matemático `all-MiniLM-L6-v2`. Embora esse modelo seja o ótimo para eficiência em CPU e compreensão de linguagem natural genérica (em inglês), os testes iniciais de recuperação de código-fonte não foram promissores. O primeiro empecilho foi que o modelo estava trazendo classes de teste ao questionar sobre uma regra de negócio. Isso aconteceu pois nessas classes de código as palavras buscadas apareciam muitas vezes, favorecendo esse arquivo na busca vetorial. A solução foi adicionar na pipeline de ingestão dos documentos um filtro dos arquivos de teste como um todo.

Os filtros servem para remover informações não relevantes para a consulta, que podem atrapalhar a relação matemática dos embedddings, como código boilerplates, imports e liceças de uso. Um problema encontrado nessa etapa era que os chunks consultados estavam tomados de linhas em branco, que eram intepretados como limites de blocos pelo LangChain, gerando dados ruidosos. Então, foi includio mais um filtro para remover as quebras de linha.

Outra situação percebida foi a diferença dos métodos de busca vetorial. O sistema começou utilizando `similarity_search` na busca de embeddings. Porém, isso fez com que muitas vezes trechos do mesmo arquivo retornassem na busca, dificultando a criação de uma resposta mais completa em relação ao fluxo, algo que atravessa vários arquivos. Por exemplo, Ao buscar algo relacionado a como fazer um busca de pets, é mais interessante que retornem trechos de arquivos de mais de uma camada, como repositório, serviço e controlador, fornecendo uma resposta completa sobre como o usuário pode fazer a busca de forma efetiva. Para resolver esse problema, o algoritmo passou a utilizar `max_marginal_relevance_search` (MMR). A ideia dessa busca é que após a primeira consulta de trechos mais relevantes, ainda é feito um filtro iterativo para manter os mais relevantes mas com uma diferença entre si.

Por fim, o maior problema encontrado nessa abordagem foi a distância semântica do prompt recebido para o código-fonte. Como citado anteriormente, os códigos hoje são apoiados por frameworks que fazem a parte mais repetitiva de determinado objetivo, englobando a lógica mais ténica. Nesse exemplo, o framework do JPA, usado para interaçao com o banco de dados, é responsável pela geração das consultas em SQL, fornecendo uma maneira de criar consultas através de interfaces no Java e um padrão de nomenclatura de métodos. Essa abstração faz com que os termos buscados pelo usuário não necessariamente sejam encontrados nesse arquivo de interface, trazendo então documentos bem menos relacionados a pergunta.

Nesse momento foi introduzida uma técnica de *expansão de query*. No momento da consulta, antes da pergunta ser repassada para a consulta aos embeddings, fazemos uma tradução de português para uma linguagem técnica, utilizando técnicas de Few-shot Promting. O prompt que obteve mais sucesso nessa tarefa consiste em determinar algumas regras de formato da resposta, e algumas instruções para a LLM encontrar conceitos chaves na pergunta que estão relacionados com o universo Spring Boot, como identificar a entidade, ação e camada relacionadas. O prompt de expansão pode ser encontrado no arquivo `configs.py`.

Também relacionado ao problema da distância semântica, foi observado que o modelo de embedding matemático utilizado estava chegando a um limite. Como ele é um modelo de linguagem de uso geral, faltam noções mais específicas de software. O modelo então foi subtituído pelo `microsoft/unixcoder-base`, um modelo específico para codificação, que é treinado com as noções de árvore sintática ao invés de somente texto, entendendo a hierarquia de classes e métodos.

## Benchmark

Para realizar os testes foi escrito um benchmark padronizado, onde perguntas previamente escritas passam pelo processo de busca de trechos semelhantes. Cada pergunta está relacionada a uma categoria e está vinculada a termos esperados na resposta. Esse teste serve para analisar se os embeddings salvos no banco vetorial estão coesos, e se o prompt expandido foi o suficiente para traduzir a pergunta para os termos esperados.

Os primeiros gráficos são utilizando o primeiro modelo, `all-MiniLM-L6-v2` antes dos filtros.

![all-MiniLM-L6-v2 hit rate antes dos filtros](./docs/all_mini_hit_rate_no_filter.png)
![all-MiniLM-L6-v2 taxa de acerto antes dos filtros](./docs/all_mini_taxa_acerto_no_filter.png)
![all-MiniLM-L6-v2 latência por etapas antes dos filtros](./docs/all_mini_latencia_etapas_no_filter.png)
![all-MiniLM-L6-v2 variância tempo antes dos filtros](./docs/all_mini_variancia_tempo_no_filter.png)

Os próximos gráficos são utilizando o primeiro modelo, `all-MiniLM-L6-v2` depois dos filtros.

![all-MiniLM-L6-v2 hit rate depois dos filtros](./docs/all_mini_hit_rate.png)
![all-MiniLM-L6-v2 taxa de acerto depois dos filtros](./docs/all_mini_taxa_acerto.png)
![all-MiniLM-L6-v2 latência por etapas depois dos filtros](./docs/all_mini_latencia_etapas.png)
![all-MiniLM-L6-v2 variância tempo depois dos filtros](./docs/all_mini_variancia_tempo.png)

Fica visível que utilizando o modelo generalista sem os filtros as taxas de acerto são bem baixas, indicando que o contexto para a busca não foi o suficiente e que não houve uma correlação entre a pergunta e os arquivos corretos. Após o filtro de arquivos de teste, lincenças, imports e quebra de linha, a taxa de acerto aumentou consideravelmente.

Os próximos gráficos são utilizando o segundo modelo, `microsoft/unixcoder-base`, já com os filtros aplicados.

![unixcode hit rate](./docs/unixcoder_hit_rate.png)
![unixcode taxa de acertos](./docs/unixcoder_taxa_acerto.png)
![unixcode latência por etapas](./docs/unixcoder_latencia_etapas.png)
![unixcode variância no tempo](./docs/unixcoder_variancia_tempo.png)

Ao trocar o modelo de embedding temos um aumento considerável nos acertos. Ainda está a baixo de um bom modelo, o que singnifica que há espaço para melhoria, ou alterando os parâmetros do MMR, o template de expansão, ou adicionando mais informaçoes ao documento armazenado no banco vetorial. Em relação aos tempos, podemos ver uma desvantagem dessa arquitetura, que é a latência na tradução da frase buscada para as palavras chave, que represantam praticamente todo o tempo consumido em cada pergunta e busca no banco vetorial.

## Demonstração de uso

Instalar as dependências com `pip install -r requirements.txt`. É recomendado o uso de ambientes virtualizados do Python para evitar conflito de versões.

Para utilizar, primeiro faça o clone do repositório. Caso queria editar algum dos parâmetros de execução, faça isso nesse momento no arquivo `configs.py`. Com os parâmetros estabelecidos, execute o script de ingestão `ingestion.py`. Aqui será criado o diretório `DB_DIR` (`db/` por padrão) para uso do ChromaDB com os dados dos embeddings de cada arquivo `.java` encontrado e filtrado. Depois, basta executar a consulta em `query.py`. Esse script irá dispor um prompt onde você digitará a sua pergunta e aguardará a resposta.

Caso queira visualizar os gráfico, é preciso executar o `benckmarck.py` e depois o `dashboard.py`. O primeiro criará um arquivo csv com o resultado do teste, e o segundo criará um diretório `charts` com os gráficos. Lembrando que todos os scripts utilizam das variáveis no `config.py`, portanto não altere depois de ter feito a ingestâo para manter a coerência.

### Parâmetros

Todos os parâmetros, com exceção dos templates, são parametrizáveis também via variável de ambiente de mesmo nome.

- DATA_DIR: Código-fonte a ser analisado (padrão `/spring-petclinic`)
- DB_DIR: Diretório do ChromaDB com embeddings (padrão `./db`)
- EMBEDDING_MODEL: Modelo que gera os embeddings (padrão `microsoft/unixcoder-base`)
- LLM_MODEL: Modelo o LLama (padrão `llama3.2`)
- CHUNK_SIZE: Tamanho do chunk (padrão `1200`)
- CHUNK_OVERLAP: Tamanho do overlap de chunk (padrão `200`)
- MMR_FETCH_K: Número de embeddings mais semelhantes a buscar (padrão `15`)
- MMR_K: Número de embeddings distintos (padrão `5`)
- SHOW_CONTEXT_MATCH: Mostrar ou não templates intermediários na consulta (padrão `True`)
- BENCHMARK_CSV: Arquivo de saída do benchmark (padrão `benchmark_results.csv`)
- CHARTS_DIR: Diretório de saída dos gráficos (padrão `./charts`)
- EXPANSION_TEMPLATE: Template que converte prompt em termos técnicos
- TEMPLATE: Template que recebe o contexto e os termos técnicos e escreve a resposta final

### Screenshots

![Ingestão](./docs/ingestion_out.png)

![Consulta](./docs/query_out.png)

## Passos para usar em produção

Para utilizar o modelo em produção seriam necessárias algumas alterações na arquitetura, que foi muito pensada em uma prova de conceito. Primeiro, precisamos adotar um banco vetorial em um servidor a parte. Precisariamos também de uma API e uma interface web para que os desenvolvedores pudessem escrever suas perguntas, no modelo chat. Essa etapa poderia ser feita junto do próprio projeto, com Python usando FastAPI, por exemplo.

Seria preciso também conteinerizar a aplicação e criar uma pipeline de CI e CD, para que seja possível subir e atualizar instâncias em um cluster Kubernetes usando Docker/Podman conforme atualizamos os scripts/API.

Com relação aos embeddings, também seria preciso uma forma de atualizar o banco vetorial conforme o código-fonte analisado é alterado. Isso poderia ser feito através de um hook do repositório atual para iniciar o processo de atualização dos embeddings.
