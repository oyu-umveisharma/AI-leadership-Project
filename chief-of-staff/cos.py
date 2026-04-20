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

  cos platform status                   Platform health, issues, agent status
  cos platform tasks [--filter open|all] List CoS task list
  cos platform resolve <id>             Mark a task resolved
  cos platform dismiss <id>             Mark a task dismissed
  cos platform add "<title>"            Add a manual task
      --desc     "description"
      --priority critical|high|medium|low
  cos platform sweep                    Run a full oversight sweep now
"""

import sys
import argparse
from pathlib import Path

# Make modules importable when run from any directory
sys.path.insert(0, str(Path(__file__).parent))

from modules import briefing, triage, meeting, decisions, followups, weekly
from modules import llm
from modules import platform as platform_mod


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

    # platform
    p_plat = sub.add_parser("platform", help="Platform oversight — health, tasks, sweep")
    plat_sub = p_plat.add_subparsers(dest="plat_command", metavar="<subcommand>")

    plat_sub.add_parser("status",  help="Platform health score and issue summary")

    p_pt = plat_sub.add_parser("tasks", help="List CoS tasks")
    p_pt.add_argument("--filter", default="open", choices=["open", "all"],
                      help="Filter tasks (default: open)")

    p_pr = plat_sub.add_parser("resolve", help="Mark a task resolved")
    p_pr.add_argument("id", help="Task ID (e.g. a1b2c3d4)")

    p_pd = plat_sub.add_parser("dismiss", help="Dismiss a task")
    p_pd.add_argument("id", help="Task ID")

    p_pa = plat_sub.add_parser("add", help="Add a manual task")
    p_pa.add_argument("title", help="Task title")
    p_pa.add_argument("--desc",     default="", help="Optional description")
    p_pa.add_argument("--priority", default="medium",
                      choices=["critical", "high", "medium", "low"],
                      help="Priority (default: medium)")

    plat_sub.add_parser("sweep", help="Run a full oversight sweep immediately")

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

    elif args.command == "platform":
        if not args.plat_command:
            print("Usage: cos platform <status|tasks|resolve|dismiss|add|sweep>")
            sys.exit(1)
        if args.plat_command == "status":
            platform_mod.status()
        elif args.plat_command == "tasks":
            platform_mod.tasks(filter_by=args.filter)
        elif args.plat_command == "resolve":
            platform_mod.resolve(args.id)
        elif args.plat_command == "dismiss":
            platform_mod.dismiss(args.id)
        elif args.plat_command == "add":
            platform_mod.add(args.title, args.desc, args.priority)
        elif args.plat_command == "sweep":
            platform_mod.sweep()


if __name__ == "__main__":
    main()
