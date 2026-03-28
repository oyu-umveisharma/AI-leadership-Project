#!/usr/bin/env python3
"""
Chief of Staff — AI Executive Assistant for the CRE Intelligence Platform
─────────────────────────────────────────────────────────────────────────
Usage:
  cos briefing                          Daily status briefing
  cos triage                            Prioritize tasks.md
  cos prep "<topic>"                    Meeting prep doc for a topic
  cos decide "<title>"                  Log an architectural decision
      --context  "why this decision"
      --options  "opt A | opt B | opt C"
      --decision "what was decided"
      --rationale "why"
  cos decisions [--n N]                 List recent decisions (default: 5)
  cos follow-up add "<item>"            Add a follow-up action item
      --owner "<name>"
      --due   "YYYY-MM-DD"
  cos follow-up list [--filter open|overdue|done|all]
  cos follow-up done <id>               Mark a follow-up complete
  cos weekly                            Weekly digest
"""

import sys
import argparse
from pathlib import Path

# Make modules importable when run from any directory
sys.path.insert(0, str(Path(__file__).parent))

from modules import briefing, triage, meeting, decisions, followups, weekly
from modules import llm


def _print_help():
    print(__doc__)


def main():
    parser = argparse.ArgumentParser(
        prog="cos",
        description="Chief of Staff — AI executive assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `cos <command> --help` for command-specific help.",
        add_help=True,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # briefing
    sub.add_parser("briefing", help="Daily repo status briefing")

    # triage
    sub.add_parser("triage", help="Prioritize tasks from tasks.md")

    # prep
    p_prep = sub.add_parser("prep", help="Meeting prep for a topic")
    p_prep.add_argument("topic", help="Topic or agenda for the meeting")

    # decide
    p_decide = sub.add_parser("decide", help="Log an architectural/product decision")
    p_decide.add_argument("title", help="Short title for the decision")
    p_decide.add_argument("--context",   default="", help="Why this decision was needed")
    p_decide.add_argument("--options",   default="", help="Options considered (separate with |)")
    p_decide.add_argument("--decision",  default="", help="What was decided")
    p_decide.add_argument("--rationale", default="", help="Why this option was chosen")

    # decisions list
    p_decisions = sub.add_parser("decisions", help="List recent decisions")
    p_decisions.add_argument("--n", type=int, default=5, help="Number of decisions to show (default: 5)")

    # follow-up
    p_fu = sub.add_parser("follow-up", help="Manage follow-up action items")
    fu_sub = p_fu.add_subparsers(dest="fu_command", metavar="<subcommand>")

    p_fu_add = fu_sub.add_parser("add", help="Add a follow-up item")
    p_fu_add.add_argument("item",    help="Description of the action item")
    p_fu_add.add_argument("--owner", default="", help="Person responsible")
    p_fu_add.add_argument("--due",   default="", help="Due date (YYYY-MM-DD)")

    p_fu_list = fu_sub.add_parser("list", help="List follow-ups")
    p_fu_list.add_argument("--filter", default="open",
                           choices=["open", "overdue", "done", "all"],
                           help="Filter by status (default: open)")

    p_fu_done = fu_sub.add_parser("done", help="Mark a follow-up as complete")
    p_fu_done.add_argument("id", type=int, help="Follow-up ID to mark done")

    # weekly
    sub.add_parser("weekly", help="Weekly progress digest")

    # ── Parse ──────────────────────────────────────────────────────────────────
    args = parser.parse_args()

    if not args.command:
        _print_help()
        sys.exit(0)

    # LLM status hint
    if llm.available():
        print("[cos] 🤖 AI mode: Groq LLM active\n", file=sys.stderr)
    else:
        print("[cos] 📋 Offline mode: no GROQ_API_KEY found (rule-based output)\n", file=sys.stderr)

    # ── Dispatch ───────────────────────────────────────────────────────────────
    if args.command == "briefing":
        briefing.run()

    elif args.command == "triage":
        triage.run()

    elif args.command == "prep":
        meeting.run(args.topic)

    elif args.command == "decide":
        if not args.title:
            print("Error: provide a decision title. Usage: cos decide \"<title>\" --decision \"...\"")
            sys.exit(1)
        decisions.add(
            title     = args.title,
            context   = args.context,
            options   = args.options,
            decision  = args.decision,
            rationale = args.rationale,
        )

    elif args.command == "decisions":
        decisions.list_decisions(n=args.n)

    elif args.command == "follow-up":
        if not args.fu_command:
            print("Usage: cos follow-up <add|list|done> [options]")
            sys.exit(1)
        if args.fu_command == "add":
            followups.add(args.item, args.owner, args.due)
        elif args.fu_command == "list":
            followups.list_followups(args.filter)
        elif args.fu_command == "done":
            followups.complete(args.id)

    elif args.command == "weekly":
        weekly.run()


if __name__ == "__main__":
    main()
