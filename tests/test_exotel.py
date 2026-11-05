import httpx

from app.providers.exotel import ExotelTelephony


def test_exotel_connect_request_is_built_per_docs():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["form"] = dict(item.split("=", 1) for item in request.content.decode().split("&"))
        return httpx.Response(200, json={"Call": {"Sid": "abc123", "Status": "queued"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    telephony = ExotelTelephony("SID1", "key", "tok", flow_url="http://my.exotel.com/SID1/exoml/start_voice/42", client=client)
    result = telephony.dial("+919999900001", "+918012345678", "https://cb.example.com/status", custom_field="7")

    assert result.accepted and result.provider_call_id == "abc123"
    assert seen["url"] == "https://api.exotel.com/v1/Accounts/SID1/Calls/connect"
    assert seen["auth"].startswith("Basic ")
    form = seen["form"]
    assert form["From"] == "%2B919999900001"          # customer, dialed first
    assert form["CallerId"] == "%2B918012345678"      # our ExoPhone
    assert form["CustomField"] == "7"
    assert "Url" in form


def test_exotel_reports_http_failure():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    telephony = ExotelTelephony("SID1", "bad", "bad", flow_url="x", client=client)
    result = telephony.dial("+919999900001", "+918012345678", "cb")
    assert not result.accepted
    assert result.reason == "http_401"
