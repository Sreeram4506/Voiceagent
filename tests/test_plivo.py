import httpx

from app.providers.plivo import PlivoTelephony


def test_plivo_call_request_is_built_correctly():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = request.content.decode()
        return httpx.Response(
            201,
            json={"api_id": "abc", "message": "call fired", "request_uuid": "uuid-123"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    telephony = PlivoTelephony("AUTH_ID", "AUTH_TOKEN", public_url="https://example.ngrok.app", client=client)
    result = telephony.dial("+919999900001", "+918065591949", "cb", custom_field="7")

    assert result.accepted and result.provider_call_id == "uuid-123"
    assert seen["url"] == "https://api.plivo.com/v1/Account/AUTH_ID/Call/"
    assert seen["auth"].startswith("Basic ")
    assert '"from": "918065591949"' in seen["body"]
    assert '"to": "919999900001"' in seen["body"]
    assert "answer/7" in seen["body"]


def test_plivo_reports_http_failure():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    telephony = PlivoTelephony("BAD", "BAD", public_url="https://x.ngrok.app", client=client)
    result = telephony.dial("+919999900001", "+918065591949", "cb")
    assert not result.accepted
    assert result.reason == "http_401"


def test_plivo_hangup_sends_delete():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        return httpx.Response(204)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    telephony = PlivoTelephony("AUTH_ID", "AUTH_TOKEN", public_url="https://x.ngrok.app", client=client)
    telephony.hangup("call-uuid-123")

    assert seen["method"] == "DELETE"
    assert "call-uuid-123" in seen["url"]
