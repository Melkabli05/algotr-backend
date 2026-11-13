from typing import Annotated

from pydantic import StringConstraints

Symbol = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_upper=True,
        pattern=r"(?i)^[A-Z0-9][A-Z0-9.]{0,9}(/[A-Z0-9]{2,10})?$",
    ),
]
