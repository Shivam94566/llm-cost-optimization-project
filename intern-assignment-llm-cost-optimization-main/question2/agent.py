from __future__ import annotations

import json
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import parse_resolution

ALLOWED_ACTIONS = {"REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"}


@dataclass
class TicketResolution:
    action: str
    kb_article: str


class SupportPipeline:
    """triage -> retrieve -> compose -> audit.

    Only `compose` calls the LLM. The other three nodes are cheap deterministic
    code, so the 200k-token help center is never sent to the model.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"))

        candidates = self._triage(order)            # code, 0 points
        articles = self._retrieve(candidates)       # code, 0 points
        draft = self._compose(thread, snapshot, articles)   # the ONLY LLM call
        return self._audit(draft, candidates)       # code, 0 points

    # ---- node 1: triage (rules on order facts, no LLM) -------------------
    def _triage(self, order: dict) -> list[str]:
        ptype = order.get("product_type")
        if ptype == "subscription":
            return ["subscription-cooling-off"]
        if order.get("license_revealed") or ptype == "digital":
            return ["digital-license-no-refund"]
        if order.get("damage_reported"):
            if order.get("customized"):
                return ["custom-items-no-replace"]
            return ["damage-replacement-7-days", "returns-within-30-days"]
        return ["returns-within-30-days"]

    # ---- node 2: retrieve (load only the 1-2 relevant articles) ----------
    def _retrieve(self, candidates: list[str]) -> str:
        known = set(article_ids())
        return "\n".join(
            f"--- {a} ---\n{load_article(a)}" for a in candidates if a in known
        )

    # ---- node 3: compose (single, small LLM call) ------------------------
    def _compose(self, thread: str, snapshot: str, articles: str) -> str:
        messages = [
            ChatMessage(
                "system",
                "ShopAssist Tier-1 router. JSON only: action, kb_article, customer_reply. "
                "action: REFUND, REPLACE, SEND_ARTICLE or ESCALATE_TIER2.\n"
                f"ARTICLES:\n{articles}",
            ),
            ChatMessage("user", f"THREAD:\n{thread}\nORDER_SNAPSHOT:{snapshot}END_SNAPSHOT"),
        ]
        return self.llm.chat("compose", messages)

    # ---- node 4: audit (validate in code, no LLM) ------------------------
    def _audit(self, draft: str, candidates: list[str]) -> TicketResolution:
        parsed = parse_resolution(draft)
        action, article = parsed.action, parsed.kb_article
        if action not in ALLOWED_ACTIONS or article not in set(article_ids()):
            # invalid model output -> safest fallback: human review
            return TicketResolution("ESCALATE_TIER2", candidates[0])
        return TicketResolution(action, article)