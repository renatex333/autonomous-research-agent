import os
import sys
import logging
import logging.config
from typing import TypedDict, List, cast
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END

# --- Configurações e Constantes ---
load_dotenv()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
IS_DEBUG = LOG_LEVEL == "DEBUG"
MAX_RETRIES = 3

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S.%f",
        },
        "detailed": {
            "format": "%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S.%f",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "detailed" if IS_DEBUG else "default",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "python_multipart": {
            "handlers": ["console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Verifica se as chaves foram carregadas
if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    logger.error("ERRO: Chaves de API não encontradas. Verifique seu arquivo .env")
    sys.exit()

# --- Definições de Classes e Funções ---
class ResearchState(TypedDict):
    """Representa o estado do grafo de pesquisa."""
    topic: str
    messages: List[BaseMessage]
    search_results: str
    decision: str
    report: str
    retries: int

def configure_tools():
    """Inicializa as ferramentas e o modelo de linguagem."""
    web_search_tool = TavilySearch(max_results=3)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return web_search_tool, llm

def search_node(state: ResearchState, web_search_tool: TavilySearch):
    """Nó para pesquisar na web usando a ferramenta Tavily."""
    logger.info("--- 🔍 NÓ DE PESQUISA ---")
    retries = state.get("retries", 0) + 1
    topic = state["topic"]
    results = web_search_tool.invoke({"query": topic})
    logger.debug("Resultados da pesquisa: %s", results)
    return {"search_results": results, "retries": retries}

def analyze_node(state: ResearchState, llm: ChatOpenAI):
    """Nó para analisar os resultados da pesquisa e decidir o próximo passo."""
    logger.info("--- 🧐 NÓ DE ANÁLISE ---")
    topic = state["topic"]
    search_results = state["search_results"]

    response = llm.invoke(f"""
        Analisando os seguintes resultados de pesquisa sobre o tópico '{topic}':
        {search_results}

        A informação é suficiente para escrever um relatório simples?
        Responda apenas com 'continue' se for suficiente, ou 'rewrite' se a pesquisa precisar ser refeita com um novo foco.
    """)

    decision = str(response.content).strip().lower()
    logger.debug("Decisão da análise: %s", decision)
    return {"decision": decision}

def write_node(state: ResearchState, llm: ChatOpenAI):
    """Nó para gerar o relatório final com base nos resultados da pesquisa."""
    logger.info("--- ✍️ NÓ DE ESCRITA ---")
    topic = state["topic"]
    search_results = state["search_results"]

    response = llm.invoke(f"""
        Com base nas seguintes informações de pesquisa sobre '{topic}':
        {search_results}

        Por favor, escreva um relatório conciso e bem estruturado sobre o tópico.
    """)

    report = response.content
    logger.debug("Relatório gerado:\n%s", report)
    return {"report": report}

def should_continue(state: ResearchState):
    """Decide se continua para a escrita ou volta para a pesquisa."""
    decision = state["decision"]
    retries = state["retries"]

    if retries > MAX_RETRIES:
        logger.warning("--- ⚠️ LIMITE DE TENTATIVAS ATINGIDO ---")
        return "continue"

    return "continue" if "continue" in decision else "rewrite"

def build_workflow(web_search_tool: TavilySearch, llm: ChatOpenAI):
    """Monta e compila o grafo de estados."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("search", lambda state: search_node(cast(ResearchState, state), web_search_tool))
    workflow.add_node("analyze", lambda state: analyze_node(cast(ResearchState, state), llm))
    workflow.add_node("write", lambda state: write_node(cast(ResearchState, state), llm))

    workflow.set_entry_point("search")
    workflow.add_edge("search", "analyze")
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "continue": "write",
            "rewrite": "search",
        },
    )
    workflow.add_edge("write", END)

    return workflow.compile()

def main():
    logger.info("🚀 Iniciando Agente de Pesquisa Autônomo...")
    topic = input("Qual tópico você gostaria de pesquisar? ")

    web_search_tool, llm = configure_tools()
    app = build_workflow(web_search_tool, llm)

    # Garante que o input inicial seja do tipo correto
    initial_state: ResearchState = {
        "topic": topic,
        "messages": [],
        "search_results": "",
        "decision": "",
        "report": "",
        "retries": 0
    }

    final_state = app.invoke(initial_state)

    logger.info("\n\n--- ✅ PESQUISA CONCLUÍDA ---")
    logger.info("Relatório Final:")
    logger.info(final_state.get("report", "Nenhum relatório foi gerado."))


# --- Execução Principal ---
if __name__ == "__main__":
    main()
