"""SupportTicketExtraction — Project 1's second extraction schema
(ARCHITECTURE.md "Two schemas"). Continues this course's running Nimbus
Cloud Storage support-ticket theme (Modules 13, 15-17, 19, 22), reusing
Module 19's exact four-category taxonomy for continuity rather than
inventing a new one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SupportTicketExtraction(BaseModel):
    category: Literal["account", "billing", "technical", "security"] | None = None
    urgency: Literal["low", "medium", "high"] | None = None
    mentioned_product: str | None = None
    customer_email: str | None = None
    summary: str | None = None
