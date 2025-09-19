# 🌿 Data Visualization for Sustainability  🌿 
### Week-Long Python & Colab Course

In this course, you’ll learn how to use **Python** and **Jupyter/Colab notebooks** to create powerful, beautiful, and insightful visualizations about the planet.

You don’t need to be a programmer — if you have a bit of stats background and a curiosity about sustainability, you’re ready. This week is all about turning real-world environmental data into visual stories that matter. Think **climate change, renewable energy, air pollution, biodiversity loss** — and how you can use code and graphics to make those issues understandable, shareable, and impactful.

You’ll work entirely in **Google Colab** — a free, browser-based platform. There’s nothing to install. You’ll use public, real-world datasets and open-source Python tools like `pandas`, `matplotlib`, `seaborn`, and `plotly`. Every day, you’ll work through an interactive notebook built around a hands-on project — one that teaches you a core technique and tells a real story about sustainability.

Each day follows a simple rhythm:  
> **Crawl** → You’ll start with guided steps and lots of support.  
> **Walk** → You’ll fill in key parts yourself with scaffolding and hints.  
> **Run** → You’ll build something visually impressive and interpret what it means.

You'll have:
- **Starter notebooks** with explanations, prompts, and partially written code
- **Solution notebooks** so you can check your work or get unstuck
- Clear instructions, compelling datasets, and space to add your own voice

By the end of the week, you’ll be able to create and share your own data-driven visual narrative — one that doesn’t just show numbers, but helps people **understand what they mean**. You’ll know how to explore datasets, choose the right type of plot, build interactive charts, and turn data into insight.

This course is about **using code as a storytelling tool** — not mastering syntax, but mastering meaning. Let’s get started.

## 📁 Course structure

Each day now lives in its own folder under `days/` with two notebooks:

| Day | Starter notebook | Solution notebook |
| --- | ---------------- | ----------------- |
| Day 1 – Climate Change & Basic Plotting | [`days/day01/notebook/day01_starter.ipynb`](days/day01/notebook/day01_starter.ipynb) | [`days/day01/solution/day01_solution.ipynb`](days/day01/solution/day01_solution.ipynb) |
| Day 2 – Fossil Fuels vs. Renewables | [`days/day02/notebook/day02_starter.ipynb`](days/day02/notebook/day02_starter.ipynb) | [`days/day02/solution/day02_solution.ipynb`](days/day02/solution/day02_solution.ipynb) |
| Day 3 – Pollution and Public Health | [`days/day03/notebook/day03_starter.ipynb`](days/day03/notebook/day03_starter.ipynb) | [`days/day03/solution/day03_solution.ipynb`](days/day03/solution/day03_solution.ipynb) |
| Day 4 – Biodiversity & Deforestation Mapping | [`days/day04/notebook/day04_starter.ipynb`](days/day04/notebook/day04_starter.ipynb) | [`days/day04/solution/day04_solution.ipynb`](days/day04/solution/day04_solution.ipynb) |
| Day 5 – Capstone: CO₂ Emissions & Global Temperature | [`days/day05/notebook/day05_starter.ipynb`](days/day05/notebook/day05_starter.ipynb) | [`days/day05/solution/day05_solution.ipynb`](days/day05/solution/day05_solution.ipynb) |

All notebooks now read from the datasets bundled in the [`data/`](data) directory so the lessons work completely offline.

## 🛠️ Maintaining starter notebooks

The instructor versions remain the single source of truth. Run the helper script before committing to regenerate the learner copies:

```bash
python tools/generate_starters.py
```

Use `python tools/generate_starters.py --check` during CI to ensure no one edits the generated notebooks by hand. A sample GitHub Actions workflow is included at [`.github/workflows/generate-notebooks.yml`](.github/workflows/generate-notebooks.yml).

The script strips any code blocks that are followed by `# --- Starter Notebook elide: ... ---` comments, replacing them with TODO prompts in the starter files. Update the accompanying message when you add new instructor-only snippets so learners get clear guidance.
