"""Small editable workspace used to evaluate the Coding Agent workflow."""


def count_nonempty_lines(text: str) -> int:
    """Return the number of lines containing non-whitespace text."""
    return sum(bool(line.strip()) for line in text.splitlines())