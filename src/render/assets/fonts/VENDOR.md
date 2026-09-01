# Vendored fonts

The page is self-contained (SPEC 2.5): it must make no external request at view
time, so the OpenLia design system's two families are vendored here and inlined
as base64 `@font-face` sources by `src/render/theme.py`.

| File | Family | Weights | Source |
|---|---|---|---|
| `Geist-variable-latin.woff2` | Geist | 100–900 (variable) | Google Fonts, latin subset |
| `IBMPlexMono-400-latin.woff2` | IBM Plex Mono | 400 | Google Fonts, latin subset |
| `IBMPlexMono-500-latin.woff2` | IBM Plex Mono | 500 | Google Fonts, latin subset |
| `IBMPlexMono-600-latin.woff2` | IBM Plex Mono | 600 | Google Fonts, latin subset |

Latin subsets only — the dashboard renders no other script, and the four files
together cost ~78 KB once base64-encoded, against SPEC 2.5's 5 MB page budget.

Both families are licensed under the SIL Open Font License 1.1.
