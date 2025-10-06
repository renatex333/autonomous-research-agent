import os
import sys
import logging
import logging.config
from typing import TypedDict, List, cast
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langchain_community.tools import ArxivQueryRun
from langchain_core.runnables import RunnableConfig
from langchain_community.utilities.arxiv import ArxivAPIWrapper

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
        "file": {
            "class": "logging.FileHandler",
            "filename": "app.log",
            "formatter": "detailed",
            "mode": "w",
        },
    },
    "loggers": {
        # Root logger (your app)
        "": {
            "handlers": ["console", "file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        # Suppress noisy modules (set to WARNING or higher)
        "watchdog": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "watchdog.observers.inotify_buffer": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "python_multipart": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "httpcore": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "openai": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        # Add other noisy modules here as needed
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Verifica se as chaves foram carregadas
if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    logger.error(
        "ERRO: Chaves de API não encontradas. Verifique seu arquivo .env")
    sys.exit()

# --- Definições de Classes e Funções ---


class ResearchState(TypedDict):
    """Representa o estado do grafo de pesquisa."""
    topic: str
    messages: List[BaseMessage]
    search_results: str
    decision: str
    report: str
    current_query: str
    retries: int
    subtopics: List[str]
    report_content: str
    tool_choice: str


def configure_tools():
    """Inicializa as ferramentas e o modelo de linguagem."""
    web_search_tool = TavilySearch(max_results=3)
    arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0, service_tier="flex")
    return web_search_tool, arxiv_tool, llm


def planner_node(state: ResearchState, llm: ChatOpenAI):
    """Nó para planejar subtemas de pesquisa com base no tópico inicial."""
    logger.info("--- 🗂️ NÓ DE PLANEJAMENTO ---")
    topic = state["topic"]

    response = llm.invoke(f"""
        Para pesquisar sobre '{topic}', divida o tópico em subtemas relevantes.
        Responda com uma lista de três subtemas, um por linha.
    """)

    subtopics = str(response.content).strip().split("\n")
    logger.info("Subtemas gerados: %s", subtopics)
    return {"subtopics": subtopics, "current_query": subtopics[0] if subtopics else "", "retries": 0}


def router_node(state: ResearchState, llm: ChatOpenAI):
    """Nó para decidir qual ferramenta usar com base na query atual."""
    logger.info("--- 🔀 NÓ DE ROTEAMENTO ---")
    current_query = state["current_query"]

    response = llm.invoke(f"""
        Você tem acesso às seguintes ferramentas:
        - TavilySearch: Para buscas gerais na web, notícias, artigos e tópicos amplos.
        - ArXiv: Para buscar artigos científicos e técnicos (papers).

        Para a query '{current_query}', qual ferramenta é a mais apropriada?
        Responda apenas com o nome da ferramenta.
    """)

    tool_choice = str(response.content).strip()
    logger.info("Ferramenta escolhida: %s", tool_choice)
    return {"tool_choice": tool_choice}


def tavily_search_node(state: ResearchState, web_search_tool: TavilySearch):
    """Nó para pesquisar na web usando a ferramenta Tavily."""
    logger.info("--- 🔍 NÓ DE PESQUISA (TAVILY) ---")
    retries = state.get("retries", 0) + 1
    topic = state["topic"]
    current_query = state["current_query"]
    results = web_search_tool.invoke(
        {"query": f"{topic}: {current_query}"}).get("results", "")
    # logger.info("Resultados da pesquisa (Tavily): %s", results)
    return {"search_results": results, "retries": retries}


def arxiv_search_node(state: ResearchState, arxiv_tool: ArxivQueryRun):
    """Nó para pesquisar no ArXiv usando a ferramenta ArxivQueryRun."""
    logger.info("--- 🔍 NÓ DE PESQUISA (ARXIV) ---")
    retries = state.get("retries", 0) + 1
    current_query = state["current_query"]
    results = arxiv_tool.run(current_query)
    # logger.info("Resultados da pesquisa (ArXiv): %s", results)
    return {"search_results": results, "retries": retries}


def analyze_node(state: ResearchState, llm: ChatOpenAI):
    """Nó para analisar os resultados da pesquisa e decidir o próximo passo."""
    logger.info("--- 🧐 NÓ DE ANÁLISE ---")
    topic = state["topic"]
    current_query = state["current_query"]
    search_results = state["search_results"]

    response = llm.invoke(f"""
        Analisando os seguintes resultados de pesquisa sobre o tópico '{topic}' com foco em '{current_query}':
        {search_results}

        A informação é suficiente para escrever um relatório simples e informativo?
        Responda apenas com 'continue' se for suficiente, ou 'rewrite' se a pesquisa precisar ser refeita com um novo foco.
    """)

    decision = str(response.content).strip().lower()
    if decision == "continue":
        _ = state["subtopics"].pop(0) if state["subtopics"] else None
        state["retries"] = 0
        state["report_content"] += f"\n{search_results}"
        if state["subtopics"]:
            state["current_query"] = state["subtopics"][0]
            decision = "continue"
        else:
            decision = "write"
    logger.info("Decisão da análise: %s", decision)
    logger.info("Subtemas restantes: %s", state["subtopics"])
    # logger.info("Conteúdo do relatório atual: %s", state["report_content"])
    return {
        "decision": decision,
        "subtopics": state["subtopics"],
        "current_query": state["current_query"],
        "retries": state["retries"],
        "report_content": state["report_content"]
    }


def refine_query_node(state: ResearchState, llm: ChatOpenAI):
    """Nó para refinar a consulta de pesquisa com base no feedback do LLM."""
    logger.info("--- 🔄 NÓ DE REFINAMENTO DE CONSULTA ---")
    topic = state["topic"]
    current_query = state["current_query"]
    previous_results = state["search_results"]
    prompt = (f"""
        A pesquisa para o tópico '{topic}' retornou resultados insatisfatórios:
        {previous_results}
        
        O prompt utilizado foi: '{current_query}'.

        Com base nesses resultados, gere um novo prompt de pesquisa, mais específica e focada,
        para obter informações melhores. Responda apenas um texto com o novo prompt com no máximo 300 caracteres.
    """)
    if state["tool_choice"] == "ArXiv":
        prompt += ("""
            Converta a seguinte intenção de pesquisa em uma query curta e técnica,
            ideal para o motor de busca do ArXiv.
            Use palavras-chave, operadores como AND/OR, e seja conciso.
        """)
    response = llm.invoke(prompt)

    new_query = str(response.content).strip()
    logger.info("Nova consulta sugerida: %s", new_query)
    return {"current_query": new_query}


def write_node(state: ResearchState, llm: ChatOpenAI):
    """Nó para gerar o relatório final com base nos resultados da pesquisa."""
    logger.info("--- ✍️ NÓ DE ESCRITA ---")
    topic = state["topic"]
    report_content = state["report_content"]

    response = llm.invoke(f"""
        Com base nas seguintes informações de pesquisa sobre '{topic}':
        {report_content}

        Por favor, escreva um relatório conciso e bem estruturado sobre o tópico.
    """)

    report = response.content
    logger.info("Relatório gerado:\n%s", report)
    return {"report": report}


def should_continue(state: ResearchState):
    """Decide se continua para a escrita ou volta para a pesquisa."""
    decision = state["decision"]
    retries = state["retries"]
    search_results = state["search_results"]

    if retries >= MAX_RETRIES:
        logger.warning("--- ⚠️ LIMITE DE TENTATIVAS ATINGIDO ---")
        _ = state["subtopics"].pop(0) if state["subtopics"] else None
        state["retries"] = 0
        state["report_content"] += f"\n{search_results}"
        if state["subtopics"]:
            state["current_query"] = state["subtopics"][0]
            return "continue"
        return "write"
    logger.info("Decisão final para o próximo passo: %s", decision)
    logger.info("Subtemas restantes: %s", state["subtopics"])
    logger.info("Tentativas atuais: %d", retries)
    logger.info("Conteúdo do relatório atual: %s", state["report_content"])
    return "continue" if "continue" in decision else "rewrite"


def build_workflow(web_search_tool: TavilySearch, arxiv_tool: ArxivQueryRun, llm: ChatOpenAI):
    """Monta e compila o grafo de estados."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", lambda state: planner_node(
        cast(ResearchState, state), llm))
    workflow.add_node("router", lambda state: router_node(
        cast(ResearchState, state), llm))
    workflow.add_node("tavily_search", lambda state: tavily_search_node(
        cast(ResearchState, state), web_search_tool))
    workflow.add_node("arxiv_search", lambda state: arxiv_search_node(
        cast(ResearchState, state), arxiv_tool))
    workflow.add_node("analyze", lambda state: analyze_node(
        cast(ResearchState, state), llm))
    workflow.add_node("write", lambda state: write_node(
        cast(ResearchState, state), llm))
    workflow.add_node("refine_query", lambda state: refine_query_node(
        cast(ResearchState, state), llm))

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["tool_choice"],
        {
            "TavilySearch": "tavily_search",
            "ArXiv": "arxiv_search",
        },
    )
    workflow.add_edge("tavily_search", "analyze")
    workflow.add_edge("arxiv_search", "analyze")
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "continue": "router",
            "rewrite": "refine_query",
            "write": "write",
        },
    )
    workflow.add_edge("refine_query", "router")
    workflow.add_edge("write", END)

    return workflow.compile()


def main():
    logger.info("🚀 Iniciando Agente de Pesquisa Autônomo...")
    topic = input("Qual tópico você gostaria de pesquisar? ")

    web_search_tool, arxiv_tool, llm = configure_tools()
    app = build_workflow(web_search_tool, arxiv_tool, llm)

    # Garante que o input inicial seja do tipo correto
    initial_state: ResearchState = {
        "topic": topic,
        "messages": [],
        "search_results": "",
        "decision": "",
        "report": "",
        "current_query": topic,
        "retries": 0,
        "subtopics": [],
        "report_content": "",
        "tool_choice": ""
    }
    config = RunnableConfig(recursion_limit=50)
    final_state = app.invoke(initial_state, config=config)

    logger.info("\n\n--- ✅ PESQUISA CONCLUÍDA ---")
    logger.info("Relatório Final:")
    logger.info(final_state.get("report", "Nenhum relatório foi gerado."))


# --- Execução Principal ---
if __name__ == "__main__":
    main()
