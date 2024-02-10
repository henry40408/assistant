import argparse
import random
from functools import lru_cache
from typing import Dict, Generator, List, Mapping

import httpx
from errbot import BotPlugin, Message, ValidationException, arg_botcmd, botcmd

KEY_VIEWED_IDS = "viewed_ids"


class LinkdingPlugin(BotPlugin):
    """Interact with Linkding."""

    def get_configuration_template(self) -> Mapping:
        return {"TOKEN": None, "URL": None}

    def check_configuration(self, configuration: Mapping) -> None:
        token, url = configuration["TOKEN"], configuration["URL"]
        if not token or not url:
            return  # reset configuration, pass
        resp = httpx.get(
            f"{url}/api/user/profile/", headers={"Authorization": f"Token {token}"}
        )
        if resp.status_code != 200:
            raise ValidationException(f"{resp.status_code} {resp.text}")

    def activate(self) -> None:
        if not self.config["TOKEN"] or not self.config["URL"]:
            raise Exception("API key or URL is required.")
        return super().activate()

    @lru_cache
    @staticmethod
    def get_bookmarks(token: str, url: str) -> List[Dict]:
        resp = httpx.get(
            f"{url}/api/bookmarks/", headers={"Authorization": f"Token {token}"}
        )
        json = resp.json()
        bookmarks = []
        for b in json["results"]:
            bookmarks.append(
                {
                    "id": b["id"],
                    "url": b["url"],
                    "title": b["title"] or b["website_title"] or "",
                    "description": b["description"] or b["website_description"] or "",
                    "date_added": b["date_added"],
                }
            )
        return bookmarks

    def get_cached_bookmarks(self, cached: bool) -> List[Dict]:
        token, url = self.config["TOKEN"], self.config["URL"]
        if not cached:
            LinkdingPlugin.get_bookmarks.cache_clear()
        return LinkdingPlugin.get_bookmarks(token, url)

    @arg_botcmd(
        "--cached",
        action=argparse.BooleanOptionalAction,
        help="Cache bookmarks",
        default=True,
    )  # type:ignore
    def bookmarks(self, msg: Message, cached: bool) -> str:
        bookmarks = self.get_cached_bookmarks(cached)
        lines = []
        for b in bookmarks:
            lines.append(f"* {b['id']}: {b['title']} {b['url']}")
        return "\n".join(lines)

    @arg_botcmd("n", type=int, help="How many bookmarks should be returned")  # type:ignore
    @arg_botcmd(
        "--cached",
        action=argparse.BooleanOptionalAction,
        help="Cache bookmarks",
        default=True,
    )  # type:ignore
    @arg_botcmd(
        "--reset",
        action=argparse.BooleanOptionalAction,
        help="Reset viewed IDs",
    )  # type:ignore
    def bookmarks_random(
        self, msg: Message, n: int, cached: bool, reset: bool
    ) -> Generator[str, None, None]:
        viewed_ids = self.get(KEY_VIEWED_IDS, [])
        if reset:
            viewed_ids.clear()
        bookmarks = [
            b for b in self.get_cached_bookmarks(cached) if b["id"] not in viewed_ids
        ]
        if not bookmarks:
            yield "*empty*"
            return
        selected = random.sample(bookmarks, k=n)
        ids = list(map(lambda b: b["id"], selected))
        for bookmark in selected:
            yield f"* {bookmark['id']}: {bookmark['title']} {bookmark['url']}"
        self[KEY_VIEWED_IDS] = viewed_ids + ids
