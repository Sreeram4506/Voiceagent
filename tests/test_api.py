def test_full_api_flow(client):
    tenant = client.post("/v1/tenants", json={"name": "Acme", "webhook_url": None}).json()
    headers = {"X-API-Key": tenant["api_key"]}

    assert client.get("/v1/me/balance", headers=headers).json()["credits"] == 0

    client.post("/v1/me/topup", json={"credits": 50}, headers=headers)
    assert client.get("/v1/me/balance", headers=headers).json()["credits"] == 50

    employee = client.post(
        "/v1/employees",
        json={"name": "Riya", "language": "hi", "script": "Qualify the lead.", "window_start": "00:00", "window_end": "23:59"},
        headers=headers,
    ).json()

    lead = client.post(
        "/v1/leads",
        json={"employee_id": employee["employee_id"], "phone_e164": "+919999900001", "name": "Asha"},
        headers=headers,
    )
    assert lead.status_code == 200

    # No ExoPhone yet, so nothing is dialed.
    dispatch = client.post("/v1/dispatch", headers=headers).json()
    assert dispatch["dialed"] == []

    assigned = client.put(
        f"/v1/employees/{employee['employee_id']}/number",
        json={"phone_number": "+918012345678"},
        headers=headers,
    )
    assert assigned.status_code == 200

    dispatch = client.post("/v1/dispatch", headers=headers).json()
    assert len(dispatch["dialed"]) == 1


def test_rejects_bad_api_key(client):
    response = client.get("/v1/me/balance", headers={"X-API-Key": "nope"})
    assert response.status_code == 401


def test_rejects_invalid_phone_number(client):
    tenant = client.post("/v1/tenants", json={"name": "Acme"}).json()
    headers = {"X-API-Key": tenant["api_key"]}
    employee = client.post("/v1/employees", json={"name": "Riya", "script": "x"}, headers=headers).json()
    bad = client.post(
        "/v1/leads",
        json={"employee_id": employee["employee_id"], "phone_e164": "9999"},
        headers=headers,
    )
    assert bad.status_code == 422


def test_tenant_cannot_read_another_tenants_employee(client):
    a = client.post("/v1/tenants", json={"name": "A"}).json()
    b = client.post("/v1/tenants", json={"name": "B"}).json()
    emp_a = client.post("/v1/employees", json={"name": "A", "script": "x"}, headers={"X-API-Key": a["api_key"]}).json()
    response = client.post(
        "/v1/leads",
        json={"employee_id": emp_a["employee_id"], "phone_e164": "+919999900001"},
        headers={"X-API-Key": b["api_key"]},
    )
    assert response.status_code == 404


def test_number_cannot_be_shared_between_agents(client):
    tenant = client.post("/v1/tenants", json={"name": "Acme"}).json()
    headers = {"X-API-Key": tenant["api_key"]}
    a = client.post("/v1/employees", json={"name": "A", "script": "x"}, headers=headers).json()
    b = client.post("/v1/employees", json={"name": "B", "script": "x"}, headers=headers).json()
    number = {"phone_number": "+918012345678"}
    assert client.put(f"/v1/employees/{a['employee_id']}/number", json=number, headers=headers).status_code == 200
    clash = client.put(f"/v1/employees/{b['employee_id']}/number", json=number, headers=headers)
    assert clash.status_code == 409


def test_number_format_is_validated(client):
    tenant = client.post("/v1/tenants", json={"name": "Acme"}).json()
    headers = {"X-API-Key": tenant["api_key"]}
    e = client.post("/v1/employees", json={"name": "A", "script": "x"}, headers=headers).json()
    bad = client.put(f"/v1/employees/{e['employee_id']}/number", json={"phone_number": "8012345678"}, headers=headers)
    assert bad.status_code == 422
