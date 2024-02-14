from typing import Generator, Mapping, Match, final

import emoji
from errbot import BotPlugin, Message, botcmd, re_botcmd
from simpleaichat import AIChat
from simpleaichat.models import ChatMessage


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
            self.log.warn("OpenAI API key is required")
            return
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
        params = {"temperature": 0.0}
        ai = AIChat(
            id=history_key,
            api_key=api_key,
            console=False,
            model="gpt-4-0125-preview",
            params=params,
        )
        if history_key in self:
            ai.sessions[history_key].messages = [
                ChatMessage.model_validate(i) for i in self[history_key]
            ]
        yield ai(match[1])
        self[history_key] = ai.get_session(id=history_key).messages
