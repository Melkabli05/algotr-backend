from alpaca.common.exceptions import APIError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from requests import RequestException

REJECTED_STATUSES = {400, 403, 404, 422}


def broker_message(exc: APIError) -> str:
    try:
        return str(exc.message)
    except (ValueError, KeyError, TypeError):
        return f"Alpaca returned {exc.status_code}"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def broker_api_error(_: Request, exc: APIError) -> JSONResponse:
        status = 422 if exc.status_code in REJECTED_STATUSES else 502
        return JSONResponse({"detail": broker_message(exc)}, status_code=status)

    @app.exception_handler(RequestException)
    async def broker_unreachable(_: Request, exc: RequestException) -> JSONResponse:
        return JSONResponse({"detail": f"Alpaca unreachable: {exc}"}, status_code=502)
