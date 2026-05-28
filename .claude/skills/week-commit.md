# Week Commit

Generate and create a commit message in the format `weekN: <description>`.

## Steps

1. Run `git diff --staged --name-only` to see what files are staged.
2. Determine the week(s):
   - Look for `weekN` in staged file names (e.g. `week1.ipynb` → week1)
   - Collect all unique week numbers found across all staged files
   - If multiple weeks appear, list them in order: `week1,week2`
   - If no week is found in file names, ask the user which week(s) this commit belongs to
3. Run `git diff --staged` to understand what actually changed.
4. Write a concise one-line description of the changes (what was added/fixed/updated — not which files).
5. Construct the message as: `weekN: <description>` (all lowercase, no period at end)
   - Single week: `week2: add HR diagram plot and distance calculation`
   - Multiple weeks: `week1,week2: add data pipeline and initial visualizations`
   - Example: `week1: fix SSL workaround in data_fetch`
6. Show the commit message to the user and ask for confirmation before committing.
7. Once confirmed, run: `git commit -m "<message>"`

## Rules
- Never include file names in the description — describe the change, not the files
- Keep the description under 72 characters total (including the `weekN: ` prefix)
- Do not add Co-Authored-By or any other trailers unless the user asks
