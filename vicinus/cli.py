from importlib.resources import files
from pathlib import Path
from vicinus.samples import list_samples, get_sample
from vicinus.version import version
import argparse
import shutil
import sys


def get_skill_command(mode: str, output_dir: str | None = None) -> None:
    """Output the agent skill."""
    skill_md = (files('vicinus') / 'SKILL.md').read_text()

    if mode == 'default':
        return print(skill_md)
    elif mode in ('agent', 'codex'):
        output_dir = '.agent/skills'
    elif mode == 'claude':
        output_dir = '.claude/skills'
    elif mode == 'cursor':
        output_dir = '.cursor/skills'
    elif mode == 'opencode':
        output_dir = '.opencode/skills'

    output_path = Path(output_dir) / 'vicinus'
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'SKILL.md').write_text(skill_md)
    print(f"Skill copied to {output_path}/SKILL.md")


def get_samples_command(mode: str, args) -> None:
    if mode == 'default':
        names = list_samples()
        for i, n in enumerate(names):
            print("-" * 5 + f" {n} " + "-" * 5)
            print(get_sample(n))
            if i + 1 < len(names):
                print('-' * 20 + '\n')
    elif mode == 'list':
        names = list_samples()
        for n in names:
            print(n)
    elif mode == 'one':
        try:
            print(get_sample(args.name))
        except:
            print("Error: could not load specified sample.", file=sys.stderr)
            exit(1)
    elif mode == 'custom':
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        names = list_samples()
        for n in names:
            with open(output_dir/n, "w") as f:
                f.write(get_sample(n))
        print(f"Done. Wrote {len(names)} sample files to {output_dir}.")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='vicinus',
        description=('CLI for the Vicinus fuzzy search package. Currently '
            'supports only an agent skill export subcommand.')
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        title='commands',
        description='Subcommands for Vicinus'
    )

    # Create skill subcommand parser
    skill_parser = argparse.ArgumentParser(add_help=False)
    skill_parser.add_argument(
        '--output', '-o',
        help='Output directory (default: print to stdout)',
        default=None
    )
    skill_parser.add_argument(
        '--agent', action='store_true',
        help='Output to .agent/skills/vicinus'
    )
    skill_parser.add_argument(
        '--claude', action='store_true',
        help='Output to .claude/skills/vicinus'
    )
    skill_parser.add_argument(
        '--codex', action='store_true',
        help='Output to .agent/skills/vicinus'
    )
    skill_parser.add_argument(
        '--cursor', action='store_true',
        help='Output to .cursor/skills/vicinus'
    )
    skill_parser.add_argument(
        '--opencode', action='store_true',
        help='Output to .opencode/skills/vicinus'
    )

    # Add samples parser
    samples_parser = argparse.ArgumentParser(add_help=False)
    samples_parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all bundled sample files',
        default=None
    )
    samples_parser.add_argument(
        '--name', '-n',
        help='Get a specific sample by file name',
        default=None
    )
    samples_parser.add_argument(
        '--output', '-o',
        help='Output directory (default: print to stdout)',
        default=None
    )

    # Add subcommands
    subparsers.add_parser(
        'skill',
        parents=[skill_parser],
        help=('skill [--output path|-O path|--agent|--claude|--codex|'
            '--cursor|--opencode]: Output the agent skill to stdout, a custom '
            'output path, or one of the supported platforms.'
        ),
        description='Output the agent skill'
    )

    subparsers.add_parser(
        'samples',
        parents=[samples_parser],
        help=('samples [--list|-l]: List bundled samples.\n'
            'samples [--name name|-n name]: Print a specific sample.\n'
            'samples [--output path|-o path]: Export all files to a specific path.'
        ),
        description="List/export bundled samples"
    )

    subparsers.add_parser(
        'version',
        help=('version: Print the current package version.'),
        description="Print the current package version."
    )

    args = parser.parse_args()

    if args.command == 'skill':
        mode = 'default'
        if args.opencode:
            mode = 'opencode'
        elif args.cursor:
            mode = 'cursor'
        elif args.codex:
            mode = 'codex'
        elif args.claude:
            mode = 'claude'
        elif args.agent:
            mode = 'agent'
        elif args.output:
            mode = 'custom'
        get_skill_command(mode, args.output)
    elif args.command == 'samples':
        mode = 'default'
        if args.list:
            mode = 'list'
        elif args.name:
            mode = 'one'
        elif args.output:
            mode = 'custom'
        get_samples_command(mode, args)
    elif args.command == 'version':
        print(version())
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
