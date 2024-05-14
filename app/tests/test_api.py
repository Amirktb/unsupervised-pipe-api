from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient) -> None:
    # Given

    # When
    response = client.get(
        "http://localhost:8001/",
    )

    # Then
    assert response.status_code == 200
