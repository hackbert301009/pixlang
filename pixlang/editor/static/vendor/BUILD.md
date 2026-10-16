# Vendored editor assets

Everything in this directory is generated. Do not edit it by hand.

| File | Source |
|------|--------|
| `cm6.js` | One minified ESM bundle of CodeMirror 6: `@codemirror/state`, `view`, `language`, `commands`, `autocomplete`, `lint` and `search`, plus `@lezer/highlight` (`tags`, `styleTags`, …) |
| `fonts/inter-latin-*.woff2` | `@fontsource/inter` (SIL Open Font License 1.1, see `fonts/inter-LICENSE.txt`) |
| `fonts/jetbrains-mono-latin-*.woff2` | `@fontsource/jetbrains-mono` (SIL Open Font License 1.1, see `fonts/jetbrains-mono-LICENSE.txt`) |

The page loads nothing from a CDN at runtime.

## Rebuild

```bash
cd pixlang/editor/vendor-src
npm ci            # installs the exact versions pinned in package-lock.json
npm run build     # writes ../static/vendor/cm6.js and ../static/vendor/fonts/*
```

To upgrade, bump the versions in `vendor-src/package.json`, run `npm install`
(which updates the lockfile), rebuild, and commit `package.json`,
`package-lock.json` and the regenerated files in this directory.
`node_modules/` is git-ignored.

## Why a single bundle

CodeMirror checks extensions with `instanceof`. If two modules each embed
their own copy of `@codemirror/state`, the editor fails with
"Unrecognized extension value in extension set". `vendor-src/entry.js`
re-exports every package from **one** esbuild bundle. Every import in the
page's import map (`@codemirror/view`, `@codemirror/state`, …) points at the
same `cm6.js` URL, so the browser loads that module once.

After an upgrade, check that npm still resolves one copy of the state package:

```bash
npm ls @codemirror/state --all   # every entry must show the same version, "deduped"
```
