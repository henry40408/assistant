import logging
from logging import Logger
from typing import Any, Dict, Optional

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, HttpUrl
from simpleaichat import AIChat
from trafilatura import extract, fetch_url

MODEL = "gpt-4-0125-preview"
MODEL_PARAMS = {"temperature": 0.0}


class Extracted(BaseModel):
    """Extracted data"""

    url: HttpUrl = Field(description="URL")
    summary: str = Field(description="Summary")


class Toolset:
    def __init__(
        self, logger: Logger, ai: AIChat, *, api_key: Optional[str] = None
    ) -> None:
        self.log = logger
        self.ai = ai

        sess = ai.get_session()
        self.tool_ai = AIChat(
            api_key=api_key,
            console=False,
            model=sess.model,
            params=sess.params,
            save_messages=False,
        )

    def summarize_url(self, query: str) -> Dict[str, Any]:
        """Summarize webpage content with URL"""

        # Extract URL from the query
        prompt = f"""
        Extract URL from the text.
        Text: <|{query}|>
        URL:
        """
        extracted: Any = self.tool_ai(prompt, output_schema=Extracted)

        url: str = extracted["url"]
        self.log.debug(f"URL: {url}")

        if not url:
            return {"context": f"Failed to extract URL from {query}"}

        # Fetch webpage content and sanitize
        downloaded = fetch_url(url)
        if not downloaded:
            return {"context": f"Failed to fetch {url}"}

        doc = BeautifulSoup(str(downloaded), "html.parser")
        sanitized = extract(downloaded)
        self.log.debug(f"Sanitized content: {sanitized}")

        # Summarize webpage content
        prompt = f"""
        Summarize the following text.
        Text:
        ###
        {sanitized}
        ###
        Summary:
        """
        extracted: Any = self.tool_ai(prompt, output_schema=Extracted)
        title = doc.title.string if doc.title and doc.title.string else ""
        self.log.debug(f"summary: {extracted['summary']}")
        return {"context": extracted["summary"], "title": title}

    @property
    def tools(self):
        return [self.summarize_url]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ai = AIChat(console=False, model=MODEL, params=MODEL_PARAMS)
    prompt = "Summarize https://openai.com/blog/new-embedding-models-and-api-updates ."
    toolset = Toolset(logging.getLogger(), ai)
    reply: Any = ai(prompt, tools=toolset.tools)
    print(reply["response"])
