from typer.testing import CliRunner

from whisperx_diarize.cli import app


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--language" in result.stdout