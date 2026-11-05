"""Plivo telephony adapter.

Outbound calls use Plivo's Make a Call API. When the callee answers, Plivo
hits our answer_url and we return XML with a <Stream> element that opens a
bidirectional audio WebSocket back to /v1/stream/{call_id}.

API reference: https://www.plivo.com/docs/voice/api/call#make-a-call
"""

import logging

import httpx

from app.providers.base import DialResult

logger = logging.getLogger(__name__)


class PlivoTelephony:
    def __init__(
        self,
        auth_id: str,
        auth_token: str,
        public_url: str,
        client: httpx.Client | None = None,
    ):
        self._base = f"https://api.plivo.com/v1/Account/{auth_id}"
        self._auth = (auth_id, auth_token)
        self._public_url = public_url.rstrip("/")
        self._client = client or httpx.Client(timeout=10.0)

    @staticmethod
    def _strip_plus(e164: str) -> str:
        return e164.lstrip("+")

    def dial(
        self, to_e164: str, from_number: str, callback_url: str, custom_field: str = ""
    ) -> DialResult:
        answer_url = f"{self._public_url}/v1/plivo/answer/{custom_field}"
        hangup_url = f"{self._public_url}/v1/plivo/status/{custom_field}"

        payload = {
            "from": self._strip_plus(from_number),
            "to": self._strip_plus(to_e164),
            "answer_url": answer_url,
            "answer_method": "POST",
            "hangup_url": hangup_url,
            "hangup_method": "POST",
            "ring_timeout": "30",
            "time_limit": "600",
        }

        logger.info("Plivo dial: to=%s answer_url=%s", to_e164, answer_url)
        response = self._client.post(
            f"{self._base}/Call/",
            auth=self._auth,
            json=payload,
        )
        if response.status_code >= 400:
            logger.error("Plivo API error %s: %s", response.status_code, response.text)
            print(f"[PLIVO ERROR] {response.status_code}: {response.text}", flush=True)
            return DialResult(
                provider_call_id="",
                accepted=False,
                reason=f"http_{response.status_code}",
            )
        data = response.json()
        request_uuid = data.get("request_uuid", "")
        return DialResult(provider_call_id=request_uuid, accepted=bool(request_uuid))

    def hangup(self, provider_call_id: str) -> None:
        self._client.delete(
            f"{self._base}/Call/{provider_call_id}/",
            auth=self._auth,
        )
