# GitHub Pages with Zensical

```sh
make setup
make docs-serve
make docs
```

`zensical.toml` defines the navigation and site configuration. `make docs` builds the static site into `site/`.

## Enable deployment

1. Push the repository and its `uv.lock` to GitHub with `main` as the default branch, or adjust the workflow branch filter and deployment condition.
2. In repository **Settings → Pages**, set **Build and deployment → Source** to **GitHub Actions**.
3. Push a documentation change or manually run the **Documentation** workflow.

Pull requests build the site without deploying. Pushes to `main` build and deploy it using the official Pages artifact and deployment actions. Documentation CI installs only the locked `docs` dependency group, avoiding Kraken and PyTorch downloads.

Set `site_url` under `[project]` in `zensical.toml` to your actual Pages URL after choosing a repository. No repository URL is hard-coded.

For selected benchmark figures, copy them into `docs/assets/` and add Markdown image references to the Results page. Generated datasets, run logs, checkpoints and raw training data remain ignored and are not part of the site artifact.

Reference: [Zensical's publishing guide](https://zensical.org/docs/publish-your-site/).
