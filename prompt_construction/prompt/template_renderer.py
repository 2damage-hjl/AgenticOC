from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


TEMPLATE_ROOT = Path(__file__).resolve().parent.parent / "templates"


@lru_cache(maxsize=1)
def _get_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_ROOT)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render(template_name: str, **context: object) -> str:
    return _get_environment().get_template(template_name).render(**context).strip()
