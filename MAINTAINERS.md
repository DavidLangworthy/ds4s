# 🛠️ Maintaining starter notebooks

The instructor versions remain the single source of truth. Run the helper script before committing to regenerate the learner copies:

```bash
python tools/generate_starters.py
```

Use `python tools/generate_starters.py --check` during CI to ensure no one edits the generated notebooks by hand. A sample GitHub Actions workflow is included at [`.github/workflows/generate-notebooks.yml`](.github/workflows/generate-notebooks.yml).

The script strips any code blocks that are followed by `# --- Starter Notebook elide: ... ---` comments, replacing them with TODO prompts in the starter files. Update the accompanying message when you add new instructor-only snippets so learners get clear guidance.
