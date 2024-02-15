from typing import Any, Generator, Mapping, Match

import emoji
from errbot import BotPlugin, Message, botcmd, re_botcmd
from simpleaichat import AIChat
from simpleaichat.models import ChatMessage
from toolset import MODEL, MODEL_PARAMS, Toolset


class Emoji:
    clock = emoji.emojize(":ten-thirty:")
    robot = emoji.emojize(":robot:")
    user = emoji.emojize(":bust_in_silhouette:")


class LLMPlugin(BotPlugin):
    """Interact with LLM."""

    def get_configuration_template(self) -> Mapping:
        return {
            "OPENAI_API_KEY": None,
        }

    def activate(self) -> None:
        if not self.config:
            raise Exception("OPENAI_API_KEY is required")
        return super().activate()

    def build_history_key(self, msg: Message):
        return f"history {msg.frm}"

    @botcmd
    def llm_history(self, msg: Message, args: str) -> Generator[str, None, None]:
        """Retrieve chat history."""
        history_key = self.build_history_key(msg)
        if history_key in self:
            final_usage = 0
            for m in self[history_key]:
                cm = ChatMessage.model_validate(m)
                icon = Emoji.user if cm.role == "user" else Emoji.robot
                yield f"* {icon} {cm.content} {Emoji.clock} {cm.received_at}"
                if cm.role == "assistant" and cm.total_length:
                    final_usage = cm.total_length
            yield f"current usage: {final_usage} token(s)"
        else:
            yield "*empty*"

    @botcmd
    def llm_history_clear(self, msg: Message, args: str) -> str:
        """Clear chat history."""
        history_key = self.build_history_key(msg)
        if history_key in self:
            del self[history_key]
        return "History successfully cleared."

    @re_botcmd(pattern=r"^\.(.+)$", prefixed=False)  # type:ignore
    def chat(self, msg: Message, match: Match):
        """Chat with LLM."""
        history_key = self.build_history_key(msg)
        api_key = self.config["OPENAI_API_KEY"]
        ai = AIChat(
            id=history_key,
            api_key=api_key,
            console=False,
            model=MODEL,
            params=MODEL_PARAMS,
        )
        toolset = Toolset(self.log, ai, api_key=api_key)
        if history_key in self:
            ai.sessions[history_key].messages = [
                ChatMessage.model_validate(i) for i in self[history_key]
            ]
        reply: Any = ai(match[1], tools=toolset.tools)
        yield reply["response"]
        self[history_key] = ai.get_session(id=history_key).messages
