import streamlit as st
from main import build_workflow, configure_tools, ResearchState, END

st.title("🤖 Agente de Pesquisa Autônomo com LangGraph")

# Inicializa o estado da sessão
if "run_search" not in st.session_state:
    st.session_state.run_search = False

# Configura ferramentas e workflow
web_search_tool, arxiv_tool, llm = configure_tools()
app = build_workflow(web_search_tool, arxiv_tool, llm)

# Entrada do tópico


def start_search():
    st.session_state.run_search = True


topic = st.text_input("Digite o tópico da sua pesquisa:",
                      on_change=start_search)

# Botão para iniciar a pesquisa
if st.button("Iniciar Pesquisa") or st.session_state.run_search:
    with st.spinner("O agente está trabalhando... Por favor, aguarde."):
        # Espaço para o log de pensamentos
        log_placeholder = st.empty()
        final_report_placeholder = st.empty()

        inputs: ResearchState = {
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

        final_output = None
        node_descriptions = {
            "planner": "Planejando subtemas de pesquisa",
            "router": "Decidindo a melhor ferramenta",
            "tavily_search": "Pesquisando na web",
            "arxiv_search": "Pesquisando no ArXiv",
            "analyze": "Analisando resultados da pesquisa",
            "refine_query": "Refinando a consulta",
            "write": "Relatório concluído"
        }

        for output in app.stream(inputs):
            log_content = ""
            for key, value in output.items():
                description = node_descriptions.get(key, f"Processando: {key}")
                log_content += f"### {description}\n"
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        log_content += f"- **{sub_key}**: {sub_value}\n"
                else:
                    log_content += f"{value}\n"
                log_content += "\n"
            log_placeholder.markdown(log_content)
            final_output = output

        # Exibe o relatório final
        if final_output and END in final_output:
            final_report = final_output[END]['report']
            final_report_placeholder.success("Pesquisa Concluída!")
            final_report_placeholder.markdown(final_report)

    # Reseta o estado de execução
    st.session_state.run_search = False
