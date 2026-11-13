"""Email the site sends: for now only the password reset link.

Mail goes out through Resend's HTTP API (https://resend.com/docs/api-reference/emails/send-email).
Nothing is sent, and the reset flow stays hidden on the website, until ``RESEND_API_KEY`` is set.
The sender must be on a domain verified in Resend (``EMAIL_FROM``), and ``SITE_URL`` is where
the link in the email opens.
"""

from __future__ import annotations

import html
import logging
import os
from dataclasses import dataclass
from typing import Protocol

RESEND_URL = "https://api.resend.com/emails"
DEFAULT_FROM = "COURT <no-reply@courtfantasy.app>"
DEFAULT_SITE = "https://courtfantasy.app"

log = logging.getLogger(__name__)


@dataclass
class Email:
    to: str
    subject: str
    text: str
    html: str


class Mailer(Protocol):
    async def send(self, email: Email) -> bool: ...


class ResendMailer:
    def __init__(self, api_key: str, sender: str, transport=None):
        self._key = api_key
        self._sender = sender
        self._transport = transport

    async def send(self, email: Email) -> bool:
        import httpx

        payload = {
            "from": self._sender,
            "to": [email.to],
            "subject": email.subject,
            "text": email.text,
            "html": email.html,
        }
        async with httpx.AsyncClient(transport=self._transport, timeout=15) as http:
            try:
                resp = await http.post(RESEND_URL, json=payload, headers={"Authorization": f"Bearer {self._key}"})
            except httpx.HTTPError as exc:
                log.warning("could not reach the email service: %s", type(exc).__name__)
                return False
        if resp.status_code >= 300:
            # Resend's error names the problem (bad key, unverified domain); never the address or link.
            log.warning("email service refused a message: %s %s", resp.status_code, resp.text[:200])
            return False
        return True


def create_mailer(transport=None) -> Mailer | None:
    key = os.getenv("RESEND_API_KEY", "").strip()
    if not key:
        return None
    return ResendMailer(key, os.getenv("EMAIL_FROM", "").strip() or DEFAULT_FROM, transport)


def site_url() -> str:
    return (os.getenv("SITE_URL", "").strip() or DEFAULT_SITE).rstrip("/")


def reset_email(to: str, link: str, minutes: int) -> Email:
    text = (
        "Someone asked to reset the password for your COURT account.\n\n"
        f"Set a new password here (the link works once, for {minutes} minutes):\n{link}\n\n"
        "If it wasn't you, ignore this email: your password stays the same."
    )
    safe = html.escape(link, quote=True)
    body = (
        '<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;font-size:16px;line-height:1.5;'
        'color:#1a1a1a;max-width:480px">'
        "<p>Someone asked to reset the password for your COURT account.</p>"
        f'<p><a href="{safe}" style="display:inline-block;padding:12px 20px;border-radius:8px;'
        'background:#1a1a1a;color:#ffffff;text-decoration:none;font-weight:600">Set a new password</a></p>'
        f'<p style="color:#555;font-size:14px">The link works once, for {minutes} minutes. '
        "If it wasn't you, ignore this email: your password stays the same.</p>"
        "</div>"
    )
    return Email(to=to, subject="Reset your COURT password", text=text, html=body)
