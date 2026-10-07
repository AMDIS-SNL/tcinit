# tcinit
Create initial condition datasets for tropical cyclone modeling with atmospheric models.

## Developer setup

```sh
pip install -e ".[dev]"
nbstripout --install --attributes .gitattributes
```

The second command wires the `nbstripout` filter into this clone's
`.git/config` so Jupyter notebook outputs are stripped on `git add`.
The `--attributes .gitattributes` flag is important: it tells
`nbstripout` to rely on the committed `.gitattributes` rules instead
of writing a duplicate (higher-precedence) copy to
`.git/info/attributes`, which would silently shadow the per-path
overrides in `.gitattributes`.

Case-study notebooks that embed PNG plots for GitHub rendering are
exempted in `.gitattributes` — add new ones there as needed.
