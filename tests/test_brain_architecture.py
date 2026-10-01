from pathlib import Path


def test_v2_core_has_no_serial_next_command():
    root = Path("src/ubique/brain")
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.glob("*.py")
    )
    assert "next_command" not in combined


def test_cli_uses_v2_runtime():
    cli = Path("src/ubique/cli.py").read_text(encoding="utf-8")
    assert "NeurocognitiveRuntime" in cli
    assert "Agent(config).run()" not in cli
