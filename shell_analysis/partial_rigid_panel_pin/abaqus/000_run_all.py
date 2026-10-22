import glob
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent


def run_command(command, *, stdin_text=None):
    print("run:", " ".join(command))
    subprocess.run(
        command,
        cwd=BASE,
        input=stdin_text,
        text=True,
        check=True,
    )


def run_python_scripts(pattern):
    for script in sorted(glob.glob(str(BASE / pattern))):
        script_path = Path(script)
        run_command([sys.executable, script_path.name])


def run_analysis_bat():
    # Abaqus may ask whether old job files should be overwritten, and the bat ends
    # with pause. Feed enough "y" lines for both prompts without requiring input.
    yes_input = "y\r\n" * 20
    run_command(["cmd", "/c", "201_run_analysis.bat"], stdin_text=yes_input)


def main():
    run_python_scripts("05_*.py")
    run_python_scripts("07_*.py")
    run_analysis_bat()
    run_command([sys.executable, "300_read_result.py"])
    print("finish")


if __name__ == "__main__":
    main()
