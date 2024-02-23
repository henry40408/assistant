import logging
import logging.config
from logging import Logger
from typing import Any, Dict, Optional

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, HttpUrl
from simpleaichat import AIChat
from trafilatura import extract, fetch_url

MODEL = "gpt-4-0125-preview"
MODEL_PARAMS = {"temperature": 0.0}


class Extracted(BaseModel):
    """Extracted information"""

    url: HttpUrl = Field(description="URL")
    summary: str = Field(description="Summary in 100 words")


class Toolset:
    def __init__(
        self,
        logger: Logger,
        ai: AIChat,
        *,
        api_key: Optional[str] = None,
        session_id=None,
    ) -> None:
        self.log = logger
        self.api_key = api_key
        self.ai = ai
        self.session = ai.get_session(session_id)  # type:ignore
        self.tool_ai = AIChat(
            api_key=self.api_key,
            console=False,
            model=self.session.model,
            params=self.session.params,
            save_messages=False,
        )

    def extract_from_messages(
        self,
        prompt: str,
        query: str,
        output_schema=None,
    ):
        messages = "\n".join([f"{m.role}: {m.content}" for m in self.session.messages])
        prompt = f"""
        {prompt}
        QUERY: `{query}`
        CONTEXT: ```
        {messages}
        ```
        MENTIONED:
        """
        return self.tool_ai(prompt, output_schema=output_schema)

    def summarize_url(self, query: str) -> Dict[str, Any]:
        """Summarize webpage content with URL"""

        # Extract URL from the query
        extracted = self.extract_from_messages(
            prompt="What's the actual URL does user mentioned in the query based on the context?",
            query=query,
            output_schema=Extracted,
        )
        url = extracted["url"]
        self.log.debug(f"URL: {url}")
        if not url:
            return {"context": f"Failed to extract URL from {query}"}

        # Fetch webpage content and sanitize
        downloaded = fetch_url(url)
        if not downloaded:
            return {"context": f"Failed to fetch {url}"}

        doc = BeautifulSoup(str(downloaded), "html.parser")
        content = extract(downloaded)
        if not content:
            return {"context": f"Content is empty: {url}"}

        self.log.debug(f"First 100 characaters of extracted content: {content[:100]}")

        # Summarize webpage content
        prompt = f"""
        Summarize the following text.
        TEXT: ```
        {content}
        ```
        SUMMARY:
        """
        extracted: Any = self.tool_ai(prompt, output_schema=Extracted)

        title = doc.title.string if doc.title and doc.title.string else ""
        self.log.debug(f"Title: {title}")

        summary = extracted["summary"]
        self.log.debug(f"Summary: {summary}")

        return {"context": summary, "title": title, "url": url}

    @property
    def tools(self):
        return [self.summarize_url]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.config.dictConfig({"version": 1, "disable_existing_loggers": True})

    ai = AIChat(console=False, model=MODEL, params=MODEL_PARAMS)
    prompt = "Summarize https://openai.com/blog/new-embedding-models-and-api-updates"
    logger = logging.getLogger(__name__)
    toolset = Toolset(logger, ai)
    reply: Any = ai(prompt, tools=toolset.tools)
    print(reply["response"])
