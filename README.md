# Agente de Pesquisa Autônomo

Este projeto implementa um agente de pesquisa autônomo construído com Python, LangChain e LangGraph. Ele é projetado para decompor tópicos complexos, planejar uma estratégia de pesquisa, utilizar múltiplas ferramentas (busca na web e arquivos acadêmicos), analisar os resultados e gerar um relatório coeso e informativo.

Este projeto demonstra uma arquitetura de agente cíclica e auto-corretiva, capaz de tomar decisões e refinar suas próprias queries para obter os melhores resultados.

## Sobre o Projeto

A pesquisa manual sobre um novo tópico pode ser demorada e ineficiente. Este agente automatiza o processo, simulando o fluxo de trabalho de um pesquisador humano: planejamento, execução, análise e síntese.

O diferencial deste projeto é o uso do **LangGraph** para criar um fluxo não-linear, onde o agente pode entrar em um ciclo de "pesquisa-análise-refinamento" até estar satisfeito com a qualidade da informação, antes de prosseguir para a escrita do relatório final.

## Principais Funcionalidades

  * **Planejamento Autônomo:** Decompõe um tópico principal em subtemas pesquisáveis.
  * **Roteamento Inteligente de Ferramentas:** Decide dinamicamente qual a melhor ferramenta (busca na web ou artigos acadêmicos) para cada subtema.
  * **Geração de Query Específica:** Traduz a intenção de pesquisa para o formato de query ideal para cada ferramenta.
  * **Ciclo de Auto-Correção:** Analisa os resultados da busca e pode decidir refinar sua pergunta e pesquisar novamente se a informação for insuficiente.
  * **Acumulação de Conhecimento:** Agrega as informações coletadas de múltiplas buscas antes de gerar o relatório final.
  * **Segurança:** Utiliza um contador de tentativas para evitar loops infinitos e gastos desnecessários de API.

## Arquitetura do Agente

O fluxo de trabalho do agente é orquestrado como um grafo de estados, permitindo ciclos e lógica condicional complexa.

```
[Início: Tópico]
       |
       v
[1. Planner Node] -> Gera subtemas
       |
       v
[2. Router Node] -> Escolhe a ferramenta (Tavily ou ArXiv)
       |
       +-----> [2a. ArXiv Query Generator] -> [3a. ArXiv Search Node] ---+
       |                                                                |
       +-----> [2b. Tavily Search Node] --------------------------------+
       |                                                                |
       v                                                                v
[4. Analyze Node] -----------------------------------------------------+
       |
       +-----> (Decisão: "rewrite") -> [5. Refine Query Node] -> Volta para o [2. Router Node]
       |
       +-----> (Decisão: "continue") -> Próximo subtema -> Volta para o [2. Router Node]
       |
       +-----> (Decisão: "write") -> [6. Write Node] -> Gera relatório
                                            |
                                            v
                                         [FIM]
```

## Tecnologias Utilizadas

  * **Linguagem:** Python 3.10+
  * **Framework de Agente:** LangChain & LangGraph
  * **Modelos de Linguagem (LLM):** OpenAI API (GPT-5-nano, GPT-4o)
  * **Ferramentas de Pesquisa:**
      * Tavily AI (Busca na Web otimizada para IA)
      * ArXiv (Repositório de artigos científicos)
  * **Gerenciamento de Dependências:** Poetry
  * **Interface de Usuário:** Streamlit

## Começando

Siga estas instruções para ter uma cópia do projeto rodando na sua máquina local.

### Instalação

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/renatex333/autonomous-research-agent
    cd autonomous-research-agent
    ```

2.  **Crie e ative um ambiente virtual:**

    ```bash
    # Cria o ambiente
    python -m venv .venv

    # Ativa o ambiente (Windows)
    .\.venv\Scripts\activate

    # Ativa o ambiente (Linux/Mac)
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    O seu projeto já tem as bibliotecas definidas por meio de um arquivo `pyproject.toml`, do Poetry.
    Instale o Poetry utilizando o pip: 

    ```bash
    pip install poetry
    ```

    Agora, para instalar as dependências, basta rodar:

    ```bash
    poetry install
    ```

4.  **Configure suas chaves de API:**

      * Crie um arquivo chamado `.env` na raiz do projeto.
      * Adicione suas chaves da OpenAI e da Tavily AI dentro dele:
        ```env
        OPENAI_API_KEY="sk-..."
        TAVILY_API_KEY="tvly-..."
        ```

## Como Usar

Com o ambiente virtual ativo e o arquivo `.env` configurado, execute o script principal:

```bash
streamlit run app.py
```

A interface de usuário se abrirá no seu navegador para que você insira o tópico de pesquisa. Sente-se e observe o agente trabalhar, mostrando cada passo do seu processo de "pensamento" no terminal.

## Próximos Passos

Este projeto é uma base sólida que pode ser expandida de várias maneiras:

  * [ ] **Adicionar Mais Ferramentas:** Integrar novas ferramentas, como Wikipedia, APIs financeiras ou de busca de notícias.
  * [ ] **Adicionar Memória:** Utilizar `ChatMessageHistory` para que o agente possa se lembrar de interações passadas em uma mesma sessão.
  * [ ] **Streaming em Tempo Real:** Implementar o método `.stream()` do LangGraph para mostrar o progresso do agente em tempo real na interface web.
  * [ ] **Otimização de Custos:** Usar modelos de LLM menores e mais baratos (ex: GPT-3.5-Turbo) para tarefas mais simples como roteamento e geração de queries.
