import threading
from typing import Any, Dict, Generator, Mapping, Match

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

    LOCK_TIMEOUT_IN_SECONDS = 10

    def __init__(self, bot, name=None):
        self.locks: Dict[str, threading.Lock] = {}
        super().__init__(bot, name)

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
            total_length = 0
            for m in self[history_key]:
                cm = ChatMessage.model_validate(m)
                icon = Emoji.user if cm.role == "user" else Emoji.robot
                yield f"* {icon} {cm.content} {Emoji.clock} {cm.received_at}"
                if cm.total_length:
                    total_length = cm.total_length
            yield f"current usage: {total_length} token(s)"
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
    def chat(self, msg: Message, match: Match) -> Generator[str, None, None]:
        """Chat with LLM."""

        history_key = self.build_history_key(msg)
        self.log.debug(f"history key {history_key}")

        api_key = self.config["OPENAI_API_KEY"]
        ai = AIChat(
            id=history_key,
            api_key=api_key,
            console=False,
            model=MODEL,
            params=MODEL_PARAMS,
        )
        toolset = Toolset(self.log, ai, api_key=api_key)

        self.locks[history_key] = self.locks.get(history_key, threading.Lock())

        self.log.debug(f"try to acquire {history_key} to interact with LLM")
        acquired = self.locks[history_key].acquire(
            blocking=True,
            timeout=self.LOCK_TIMEOUT_IN_SECONDS,
        )

        if acquired:
            self.log.debug(
                f"{history_key} acquired to interact with LLM for {self.LOCK_TIMEOUT_IN_SECONDS}s"
            )
            try:
                with self.mutable(history_key, []) as messages:
                    if history_key in self:
                        ai.sessions[history_key].messages = [
                            ChatMessage.model_validate(i)
                            for i in messages  # type:ignore
                        ]
                    reply: Any = ai(match[1], tools=toolset.tools)
                    yield reply["response"]
                    messages[:] = ai.get_session(id=history_key).messages  # type:ignore
            finally:
                self.locks[history_key].release()
                self.log.debug(f"{history_key} released to interact with LLM")
        else:
            yield f"cannot acquire resource to interact with LLM in {self.LOCK_TIMEOUT_IN_SECONDS}s"
