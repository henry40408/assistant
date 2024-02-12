from functools import lru_cache
from typing import Mapping, Match

from errbot import BotPlugin, Message, arg_botcmd, botcmd, re_botcmd
from llama_index.agent import ReActAgent
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.indices import DocumentSummaryIndex, VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.readers import SimpleWebPageReader
from llama_index.tools import FunctionTool


@lru_cache
def summary_engine_from_url(url: str) -> BaseQueryEngine:
    loader = SimpleWebPageReader(html_to_text=True)
    documents = loader.load_data(urls=[url])
    index = DocumentSummaryIndex.from_documents(documents)
    return index.as_query_engine()


@lru_cache
def vector_engine_from_url(url: str) -> BaseQueryEngine:
    loader = SimpleWebPageReader(html_to_text=True)
    documents = loader.load_data(urls=[url])
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()


@lru_cache
def summarize_tool(url: str) -> str:
    """Useful to summarize a webpage with a URL."""
    query_engine = summary_engine_from_url(url)
    resp = query_engine.query("What's summary of this text collection?")
    return str(resp)


@lru_cache
def rag_tool(url: str, question: str) -> str:
    """Useful to answer a question from the webpage with a URL."""
    query_engine = vector_engine_from_url(url)
    resp = query_engine.query(question)
    return str(resp)


TOOLS = [
    FunctionTool.from_defaults(fn=rag_tool),
    FunctionTool.from_defaults(fn=summarize_tool),
]


class LLMPlugin(BotPlugin):
    """Interact with LLM."""

    def get_configuration_template(self) -> Mapping:
        return {
            "OPENAI_API_KEY": None,
        }

    def activate(self) -> None:
        if not self.config:
            self.log.warn("OpenAI API key is required")
            return
        return super().activate()

    def build_history_key(self, msg: Message):
        return f"history {msg.frm}"

    @botcmd
    def llm_history(self, msg: Message, args: str):
        """Retrieve chat history."""
        history_key = self.build_history_key(msg)
        chat_history = self.get(history_key, [])
        if not chat_history:
            yield "*empty*"
        else:
            for m in chat_history:
                yield f"* {m}"

    @botcmd
    def llm_history_clear(self, msg: Message, args: str):
        """Clear chat history."""
        history_key = self.build_history_key(msg)
        if history_key in self:
            del self[history_key]
        return "Chat history cleared."

    @arg_botcmd("n", type=int, help="How many messages should be removed.")  # type:ignore
    def llm_history_pop(self, msg: Message, n: int):
        """Remove last N messages from chat history."""
        history_key = self.build_history_key(msg)
        with self.mutable(history_key, []) as chat_history:
            for _ in range(n):
                if chat_history:
                    yield f"Removed: {chat_history.pop()}"

    @re_botcmd(pattern=r"^\.(.+)$", prefixed=False)  # type:ignore
    def chat(self, msg: Message, match: Match):
        """Chat with LLM."""
        api_key = self.config["OPENAI_API_KEY"]
        llm = OpenAI("gpt-4", api_key=api_key, temperature=0.0)
        history_key = self.build_history_key(msg)
        chat_history = self.get(history_key, [])
        agent = ReActAgent.from_tools(
            tools=TOOLS,  # type:ignore
            llm=llm,
            chat_history=chat_history,
            verbose=True,
        )
        yield agent.chat(match[1]).response
        self[history_key] = agent.memory.get()
