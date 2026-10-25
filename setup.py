from setuptools import setup, Extension                 # before distutils: 3.12+ only has setuptools' copy
from distutils.command.clean import clean as CleanCommand
from pathlib import Path                                # Cython is imported in build(): clean runs without it
import hashlib, json
import shutil, sys
import sysconfig, time

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
    'commands/psd_analyzer.py',         # facade; the four below hold its code (same directives, same semantics)
    'commands/card_models.py',
    'commands/card_edits.py',
    'commands/psd_parser.py',
    'commands/repo_traverser.py',
    'commands/query_card.py',
    'commands/management.py',           # Complex async operations with discord.py
    'commands/music_player.py',         # ThreadPoolExecutor with Process/Queue-like issues
]

# Compiler directives for optimization
COMPILER_DIRECTIVES = {
    'language_level': '3',           # Python 3
    'boundscheck': True,             # a bad index raises, never crashes (owner, 2026-10-23)
    'wraparound': True,              # negative indexes as in Python
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


STAMPS = Path('build/ext-stamps.json')      # module -> fingerprint of the source its .so was built from
JOBS = 4                                    # parallel cythonize and C compiles


def module_name(file):
    return file.replace('/', '.').replace('\\', '.')[:-3]


def fingerprint(file):
    """What a module's .so depends on: its source, the directives, the flags, Cython and Python."""
    import Cython
    h = hashlib.sha256(Path(file).read_bytes())
    h.update(repr((sorted(COMPILER_DIRECTIVES.items()), compile_args(), Cython.__version__, sys.version,
                   sysconfig.get_config_var('EXT_SUFFIX'))).encode())
    return h.hexdigest()


def compile_args():
    return ['-O3'] if sys.platform != 'win32' else ['/O2']


def built_path(file):
    return Path(file[:-3] + sysconfig.get_config_var('EXT_SUFFIX'))


def remove_orphans(files):
    """Delete every compiled module that is not a current INCLUDE_FILES build (other ABI, dropped module)."""
    keep = {built_path(f) for f in files}
    for pattern in ('*.so', '*.pyd'):
        for f in Path('.').rglob(pattern):
            if f in keep or any(part.startswith('.') or part == 'build' for part in f.parts[:-1]):
                continue
            remove_file(f)


def stale_files(files, stamps):
    """The modules whose .so is missing or was built from other source (or settings)."""
    return [f for f in files if not built_path(f).exists() or stamps.get(f) != fingerprint(f)]


def load_stamps():
    try:
        return json.loads(STAMPS.read_text())
    except (OSError, ValueError):
        return {}


def build(files):
    """Rebuild only the stale modules, JOBS at a time. A stale .so is deleted first, so a failed
    build leaves the .py to run, never an old .so next to a new .py."""
    if '--inplace' not in sys.argv and '-i' not in sys.argv:
        sys.exit("Use: python setup.py build_ext --inplace")
    remove_orphans(files)
    stamps = load_stamps()
    stale = stale_files(files, stamps)
    print(f"\n{len(files) - len(stale)} of {len(files)} compiled module(s) up to date.")
    if not stale:
        return
    print(f"Rebuilding {len(stale)}:")
    for f in stale:
        print(f"  • {f}")
        stamps.pop(f, None)
        if built_path(f).exists():
            remove_file(built_path(f))
    print()

    from Cython.Build import cythonize
    extensions = [Extension(module_name(f), [f], extra_compile_args=compile_args()) for f in stale]
    ext_modules = cythonize(extensions, compiler_directives=COMPILER_DIRECTIVES, nthreads=JOBS, force=True)
    setup(name='AutoCroissant', version='1.0', packages=[], ext_modules=ext_modules, cmdclass={'clean': CleanAll},
          options={'build_ext': {'parallel': JOBS, 'force': True}})

    for f in stale:
        if built_path(f).exists():
            stamps[f] = fingerprint(f)
    STAMPS.parent.mkdir(exist_ok=True)
    STAMPS.write_text(json.dumps(stamps, indent=1, sort_keys=True))
    missing = [f for f in stale if f not in stamps]
    if missing:
        sys.exit(f"Not built: {', '.join(missing)}")


class CleanAll(CleanCommand):
    """Custom 'python setup.py clean' command."""
    def run(self):
        clean_artifacts()
        super().run()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # needed for Windows/macOS spawn method

    files = [f for f in INCLUDE_FILES if Path(f).exists()]
    for f in sorted(set(INCLUDE_FILES) - set(files)):
        print(f"  ⚠ Skipping (not found): {f}")
    if not files:
        sys.exit("No files to compile! Check INCLUDE_FILES paths.")

    start = time.monotonic()
    if 'build_ext' in sys.argv:
        build(files)
    else:
        setup(name='AutoCroissant', version='1.0', packages=[], cmdclass={'clean': CleanAll})
    print(f"\n✓ Done in {time.monotonic() - start:.1f} s")
