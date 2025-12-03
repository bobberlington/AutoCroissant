from distutils.command.clean import clean as CleanCommand
from Cython.Build import cythonize
from pathlib import Path
from setuptools import setup, Extension
import shutil
import sys

# Files that should NOT be compiled (will cause issues)
EXCLUDE_FILES = {
    '__init__.py',
    'setup.py',
    'config.py',                        # Configuration file
    'global_config.py',                 # Configuration file
    'main.py',                          # Entry point - keep as Python
    'commands/__init__.py',
    'commands/update_bot.py',           # Uses execv/execl for restart
    'commands/utils.py',                # Core utilities, decorators, dynamic behavior
}

# Files that CAN be compiled safely
INCLUDE_FILES = [
    'commands/analytics.py',
    'commands/frankenstein.py',
    'commands/help.py',
    'commands/psd_analyzer.py',
    'commands/query_card.py',
    'commands/management.py',           # Complex async operations with discord.py
    'commands/music_player.py',         # ThreadPoolExecutor with Process/Queue-like issues
    'commands/diffusion.py',            # PyTorch callbacks and dynamic function registration
]

# Compiler directives for optimization
COMPILER_DIRECTIVES = {
    'language_level': '3',           # Python 3
    'boundscheck': False,            # Disable array bounds checking (faster)
    'wraparound': False,             # Disable negative indexing (faster)
    'initializedcheck': False,       # Disable uninitialized variable checks
    'cdivision': True,               # Use C division (faster but different for negatives)
    'embedsignature': True,          # Include function signatures in docstrings
    'infer_types': True,             # Infer types automatically (helps optimization)
    'overflowcheck': False,          # Disable integer overflow checking (faster)
    # Profile-guided optimization (optional, comment out if issues)
    'profile': False,                # Disable profiling for production
    'linetrace': False,              # Disable line tracing for production
}


def remove_file(path):
    try:
        path.unlink()
        print(f"  - Deleted: {path}")
    except Exception as e:
        print(f"  ! Could not delete {path}: {e}")


def remove_dir(path):
    try:
        shutil.rmtree(path)
        print(f"  - Deleted directory: {path}")
    except Exception as e:
        print(f"  ! Could not delete directory {path}: {e}")


def clean_artifacts():
    print("\nRemoving compiled artifacts...\n" + "-" * 60)

    # Files to delete everywhere
    patterns = [
        "*.c",
        "*.cpp",
        "*.so",
        "*.pyd",
        "*.pyc",
        "*.html",
    ]

    removed = 0

    for pattern in patterns:
        for file in Path('.').rglob(pattern):
            try:
                file.unlink()
                print(f"  - deleted {file}")
                removed += 1
            except Exception as e:
                print(f"  ! could not delete {file}: {e}")

    # Remove all __pycache__ directories
    for folder in Path('.').rglob("__pycache__"):
        try:
            shutil.rmtree(folder)
            print(f"  - removed {folder}")
            removed += 1
        except Exception as e:
            print(f"  ! could not remove {folder}: {e}")

    # Remove build/ directory
    build_dir = Path("build")
    if build_dir.exists():
        try:
            shutil.rmtree(build_dir)
            print("  - removed build/")
            removed += 1
        except Exception as e:
            print(f"  ! could not remove build/: {e}")

    if removed == 0:
        print("  Nothing to clean.")

    print("-" * 60 + "\n")


def get_python_files():
    """Find all .py files that should be compiled."""
    files = []

    # Add specific include files that exist
    for file_path in INCLUDE_FILES:
        if Path(file_path).exists():
            files.append(file_path)
            print(f"  ✓ Including: {file_path}")
        else:
            print(f"  ⚠ Skipping (not found): {file_path}")

    return files


class CleanAll(CleanCommand):
    """Custom 'python setup.py clean' command."""
    def run(self):
        clean_artifacts()
        super().run()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # needed for Windows/macOS spawn method

    clean_artifacts()

    py_files = get_python_files()

    print("=" * 60)
    print("Cython Compilation Setup - Discord Bot")
    print("=" * 60)
    print(f"\nCompiling {len(py_files)} file(s):\n")

    for f in py_files:
        print(f"  • {f}")

    print("\n" + "=" * 60 + "\n")

    if not py_files:
        print("No files to compile! Check INCLUDE_FILES paths.")
        sys.exit(1)

    extensions = [
        Extension(
            file.replace('/', '.').replace('\\', '.').replace('.py', ''),
            [file],
            extra_compile_args=['-O3'] if sys.platform != 'win32' else ['/O2'],
        )
        for file in py_files
    ]

    ext_modules = cythonize(
        extensions,
        compiler_directives=COMPILER_DIRECTIVES,
        nthreads=4,
    )

    setup(
        name='AutoCroissant',
        version='1.0',
        ext_modules=ext_modules,
        cmdclass={'clean': CleanAll},
    )

    print("\n" + "=" * 60)
    print("✓ Compilation complete!")
    print("=" * 60)
