"""Exotel telephony adapter.

Outbound calls use Exotel's "Connect to flow" endpoint: Exotel dials the customer (From),
shows our ExoPhone as caller ID (CallerId), and once answered runs the flow at Url. The flow
contains a stream applet that opens the media WebSocket to /v1/stream/{call_id}.

Per the Exotel API reference (Connect to flow):
  POST https://api.exotel.com/v1/Accounts/{sid}/Calls/connect
  auth: api_key:api_token (HTTP basic)
  fields: From, CallerId, Url, StatusCallback, CustomField, TimeLimit, TimeOut
"""

import httpx

from app.providers.base import DialResult


class ExotelTelephony:
    def __init__(
        self,
        account_sid: str,
        api_key: str,
        api_token: str,
        flow_url: str,
        subdomain: str = "api.exotel.com",
        client: httpx.Client | None = None,
    ):
        self._base = f"https://{subdomain}/v1/Accounts/{account_sid}"
        self._auth = (api_key, api_token)
        self._flow_url = flow_url
        self._client = client or httpx.Client(timeout=10.0)

    def dial(self, to_e164: str, from_number: str, callback_url: str, custom_field: str = "") -> DialResult:
        """to_e164 is the customer. from_number is our ExoPhone, shown as caller ID."""
        form = {
            "From": to_e164,
            "CallerId": from_number,
            "Url": self._flow_url,
            "StatusCallback": callback_url,
            "StatusCallbackEvents": "terminal",
            "CallType": "trans",
            "TimeLimit": "600",
        }
        if custom_field:
            form["CustomField"] = custom_field

        response = self._client.post(f"{self._base}/Calls/connect", auth=self._auth, data=form)
        if response.status_code >= 400:
            return DialResult(provider_call_id="", accepted=False, reason=f"http_{response.status_code}")
        call = response.json().get("Call", {})
        sid = str(call.get("Sid", ""))
        return DialResult(provider_call_id=sid, accepted=bool(sid))

    def hangup(self, provider_call_id: str) -> None:
        # Hanging up a live call is done through the call resource; confirm the exact call-update
        # request with Exotel before relying on this in production.
        self._client.post(f"{self._base}/Calls/{provider_call_id}", auth=self._auth, data={"Status": "completed"})
