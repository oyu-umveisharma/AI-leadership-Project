# Chief of Staff — AI Executive Assistant

A CLI tool that acts as your executive assistant for the CRE Intelligence Platform. Manages briefings, task triage, meeting prep, decisions, follow-ups, and weekly digests — all in plain markdown files with optional AI enhancement via Groq.

## Setup

```bash
# From the repo root
chmod +x chief-of-staff/cos

# Option A — run directly
./chief-of-staff/cos briefing

# Option B — add to PATH for global `cos` command
echo 'export PATH="$PATH:'"$(pwd)/chief-of-staff"'"' >> ~/.zshrc
source ~/.zshrc
cos briefing
```

**Optional — AI mode** (uses Groq, already in requirements.txt):
```bash
echo "GROQ_API_KEY=your_key_here" >> .env
```
Without a key, all commands work in rule-based offline mode.

---

## Commands

### `cos briefing`
Daily status report: recent commits, open PRs, TODO/FIXME scan, stale branches, agent cache health, sync agent status.

```bash
cos briefing
```

---

### `cos triage`
Reads `chief-of-staff/state/tasks.md`, ranks open tasks by urgency and impact, flags blockers and dependencies, suggests what to work on next.

```bash
cos triage
```

**Task format** in `tasks.md`:
```markdown
- [ ] Description of task
- [ ] [BLOCKER] Something blocking other work
- [ ] [BUG] A bug that needs fixing [OWNER: Alice] [DUE: 2026-04-01]
- [ ] [DEPENDS: other task] Task with a dependency
```

---

### `cos prep "<topic>"`
Searches the codebase, recent commits, and open tasks for context related to the topic. Generates a meeting prep doc with talking points.

```bash
cos prep "pricing agent performance"
cos prep "RSS news pipeline"
cos prep "Q2 roadmap"
```

Saved to `chief-of-staff/state/prep_<topic>.md`.

---

### `cos decide "<title>"`
Logs an architectural or product decision to `chief-of-staff/state/decisions.md`.

```bash
cos decide "Use file-based JSON cache instead of database" \
  --context "We need persistence across Streamlit reruns without infrastructure overhead" \
  --options "SQLite | Redis | File-based JSON" \
  --decision "File-based JSON in /cache directory" \
  --rationale "Zero dependencies, trivially inspectable, sufficient for 5-agent update frequency"
```

### `cos decisions [--n N]`
List the most recent N decisions (default: 5).

```bash
cos decisions
cos decisions --n 10
```

---

### `cos follow-up add "<item>"`
Adds a tracked action item to `chief-of-staff/state/follow-ups.md`.

```bash
cos follow-up add "Evaluate LoopNet API pricing" --owner "Aayman" --due "2026-04-07"
cos follow-up add "Set up staging environment" --owner "Team" --due "2026-04-01"
```

### `cos follow-up list`
Lists follow-ups. Filter options: `open` (default), `overdue`, `done`, `all`.

```bash
cos follow-up list
cos follow-up list --filter overdue
cos follow-up list --filter all
```

### `cos follow-up done <id>`
Marks a follow-up as complete.

```bash
cos follow-up done 3
```

---

### `cos weekly`
Generates a weekly digest: what shipped (commits), files changed, open tasks, blockers, overdue follow-ups, agent health, and next week's focus.

```bash
cos weekly
```

Saved to `chief-of-staff/state/weekly_YYYY-MM-DD.md`.

---

## State Files

All persistent state lives in `chief-of-staff/state/` as plain markdown:

| File | Purpose | Managed by |
|------|---------|-----------|
| `tasks.md` | Open task list — edit manually | You |
| `decisions.md` | Decision log | `cos decide` |
| `follow-ups.md` | Action item tracker | `cos follow-up` |
| `last_briefing.md` | Most recent briefing output | `cos briefing` |
| `last_triage.md` | Most recent triage output | `cos triage` |
| `weekly_YYYY-MM-DD.md` | Weekly digest archives | `cos weekly` |
| `prep_<topic>.md` | Meeting prep docs | `cos prep` |

---

## AI Mode vs Offline Mode

| Feature | Offline | AI Mode (Groq) |
|---------|---------|----------------|
| Briefing | ✅ Full | + Commit summary |
| Triage | ✅ Score-ranked | + Strategic recommendation |
| Meeting Prep | ✅ Raw context | + Talking points + key questions |
| Weekly | ✅ Full | + Executive summary + next week focus |
| Decisions / Follow-ups | ✅ Full | ✅ Same |

---

## Project Structure

```
chief-of-staff/
├── cos              Shell wrapper (add to PATH)
├── cos.py           CLI entry point (argparse dispatcher)
├── modules/
│   ├── briefing.py  Daily briefing
│   ├── triage.py    Task prioritization
│   ├── meeting.py   Meeting prep
│   ├── decisions.py Decision log
│   ├── followups.py Follow-up tracker
│   ├── weekly.py    Weekly digest
│   └── llm.py       Groq client wrapper (optional)
└── state/
    ├── tasks.md         Edit this to manage your tasks
    ├── decisions.md     Auto-maintained decision log
    └── follow-ups.md    Auto-maintained follow-up tracker
```

---

*Chief of Staff — Built for CRE Intelligence Platform | MGMT 690 AI Leadership*
