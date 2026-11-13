from datetime import date, datetime

import sqlalchemy as sa
from fastapi import APIRouter
from pydantic import BaseModel

from app.core import data_quality
from app.core.database import transaction
from app.core.tables import data_quality as findings_table

router = APIRouter(prefix="/data-quality", tags=["data-quality"])


class FindingRow(BaseModel):
    id: int
    time: datetime
    source: str
    symbol: str
    session: date
    quality_flags: int
    reason_code: str
    detail: str
    digest: str


class Report(BaseModel):
    verified: bool
    findings: list[FindingRow]


@router.get("")
def read_findings(limit: int = 200) -> Report:
    """The latest data quality findings and whether their hash chain is intact."""
    with transaction() as connection:
        rows = connection.execute(
            sa.select(findings_table).order_by(findings_table.c.id.desc()).limit(limit)
        )
        found = [FindingRow.model_validate(row._mapping) for row in rows]
        return Report(verified=data_quality.verify(connection), findings=found)
