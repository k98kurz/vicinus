from importlib import resources


def list_samples() -> list[str]:
    """Returns the list of bundled sample file names."""
    try:
        directory = resources.files("vicinus").joinpath("samples")
        return [
            str(f.name)
            for f in directory.iterdir()
            if f.is_file() and f.name[-3:] in ('.md', 'txt')
        ]
    except FileNotFoundError:
        return []

def get_sample(name: str) -> str:
    """Reads and returns the content of the named sample. Raises
        FileNotFoundError for an invalid name.
    """
    resource_path = f"samples/{name}"
    path = resources.files("vicinus").joinpath(resource_path)
    with path.open("r", encoding="utf-8") as f:
        return f.read()
