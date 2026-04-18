import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from configs import *

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

def hit_rate(df):
    # Hit Rate por Categoria (Barras)
    plt.figure(figsize=(8, 6))
    
    real_hit_rate = df.groupby('Category')['Hit'].mean() * 100
    categorias = real_hit_rate.index.tolist()
    dados_comparativos = []
    
    for cat in categorias:
        hit_real = real_hit_rate[cat]
        dados_comparativos.append({'Categoria': cat, 'Modelo': EMBEDDING_MODEL, 'Hit Rate (%)': hit_real})

    df_plot1 = pd.DataFrame(dados_comparativos)
    sns.barplot(data=df_plot1, x='Categoria', y='Hit Rate (%)', hue='Modelo')
    
    plt.title("Acertos por categoria", fontweight='bold', pad=15)
    plt.ylim(0, 110)
    plt.ylabel("Precisão (Hit Rate %)")
    plt.xlabel("Domínio (Categoria)")
    plt.xticks(rotation=45, ha='right')

    filename = CHARTS_DIR + "/hit_rate.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {filename}")


def latency(df):
    # Profiling de Latência (Barras Empilhadas)
    plt.figure(figsize=(10, 6))
    
    perguntas = [q[:10] + '...' if len(q) > 10 else q for q in df['Question']]
    t_busca = df['T_Search'].values
    t_expansao = df['T_Expansion'].values
    
    plt.bar(perguntas, t_busca, label='Busca Vetorial (Chroma)', color='#2ca02c')
    plt.bar(perguntas, t_expansao, bottom=t_busca, label='Expansão da Query (Ollama)', color='#ff7f0e')
    
    plt.title(f"Gargalos de Latência por Etapa (LLM: {LLM_MODEL})", fontweight='bold', pad=15)
    plt.ylabel("Tempo de Processamento (segundos)")
    plt.xlabel("Casos de Teste")
    plt.xticks(rotation=45, ha='right')
    
    filename = CHARTS_DIR + "/latencia_etapas.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {filename}")


def accuracy(df):
    # Taxa de Acerto Global (Pizza)
    plt.figure(figsize=(6, 6))
    
    taxa_acerto_total = df['Hit'].mean() * 100
    erros = 100 - taxa_acerto_total
    
    plt.pie([taxa_acerto_total, erros], labels=['Acertos (Top 5)', 'Falhas'], 
            autopct='%1.1f%%', startangle=90, colors=['#2ca02c', '#d62728'], explode=(0.05, 0),
            textprops={'fontsize': 12, 'weight': 'bold'})
    
    plt.title(f"Taxa de acerto (K={MMR_K})", fontweight='bold', pad=15)
    
    filename = CHARTS_DIR + "/taxa_acerto.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {filename}")


def latency_variance(df):
    # Variância de Latência (Boxplot)
    plt.figure(figsize=(8, 6))
    
    sns.boxplot(data=df, x='Category', y='Total Time (s)', palette="Set2")
    
    plt.title("Variância de Latência por Domínio", fontweight='bold', pad=15)
    plt.ylabel("Tempo Total de Resposta (segundos)")
    plt.xlabel("Domínio (Categoria)")
    plt.xticks(rotation=45, ha='right')
    
    filename = CHARTS_DIR + "/variancia_tempo.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {filename}")

def plot_charts():
    print(f"Iniciando criação dos dashboards")
    print("="*80)
    print(f"Banchmark CSV: {BENCHMARK_CSV}")
    print("="*80)

    if not os.path.exists(BENCHMARK_CSV):
        print(f"❌ ERRO: Arquivo '{BENCHMARK_CSV}' não encontrado.")
        return

    if not os.path.exists(CHARTS_DIR):
        os.mkdir(CHARTS_DIR)

    print("Lendo dados do CSV e gerando gráficos individuais...")
    df = pd.read_csv(BENCHMARK_CSV)

    hit_rate(df)
    latency(df)
    accuracy(df)
    latency_variance(df)

    print("\nTodos os gráficos foram gerados e salvos com sucesso na pasta atual!")

if __name__ == "__main__":
    plot_charts()
