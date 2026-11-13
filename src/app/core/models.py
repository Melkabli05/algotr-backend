"""Fetches the pinned FinBERT and Kronos weights in the background when they are missing."""

import asyncio
import logging
import shutil
from pathlib import Path

from app.core import sidecar

logger = logging.getLogger(__name__)

FETCHERS = {"finbert": "fetch_finbert.py", "kronos": "fetch_kronos.py"}
TIMEOUT_SECONDS = 3600


def missing(folders: dict[str, Path]) -> list[str]:
    return [name for name, folder in folders.items() if not (folder / "revision.json").exists()]


async def run(command: list[str]) -> bool:
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), TIMEOUT_SECONDS)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    text = output.decode(errors="replace").strip()
    if process.returncode != 0:
        logger.error("%s failed: %s", Path(command[0]).name, text[-2000:])
        return False
    logger.info("%s", text.splitlines()[-1] if text else Path(command[0]).name)
    return True


async def install_sidecar() -> bool:
    uv = shutil.which("uv")
    if uv is None:
        logger.warning("The FinRL environment is missing and uv is not installed to build it")
        return False
    python = str(sidecar.sidecar_python())
    requirements = str(sidecar.SIDECAR / "requirements.txt")
    logger.info("Building the FinRL environment")
    return await run(
        [uv, "venv", "--python", "3.13", "--allow-existing", str(sidecar.SIDECAR / ".venv")]
    ) and await run([uv, "pip", "install", "--python", python, "-r", requirements])


async def ensure(folders: dict[str, Path]) -> None:
    """Builds the sidecar environment if needed, then fetches each model that is not there yet."""
    names = missing(folders)
    if not names:
        return
    if not sidecar.available() and not await install_sidecar():
        return
    for name in names:
        logger.info("Fetching %s", name)
        script = str(sidecar.SIDECAR / FETCHERS[name])
        await run([str(sidecar.sidecar_python()), script, str(folders[name])])
