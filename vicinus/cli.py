import argparse
import shutil
from pathlib import Path
from importlib.resources import files


def get_skill_command(mode: str, output_dir: str | None = None) -> None:
    """Output the agent skill."""
    skill_md = (files('vicinus') / 'SKILL.md').read_text()

    if mode == 'default':
        return print(skill_md)
    elif mode in ('agent', 'codex'):
        output_dir = ...
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

    # Add subcommands
    subparsers.add_parser(
        'skill',
        parents=[skill_parser],
        help=('skill [--output path|-O path|--agent|--claude|--codex|'
            '--cursor|--opencode]: Output the agent skill to stdout, a custom '
            'output path, or one of the supported platforms.'),
        description=f'Output the agent skill'
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
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
