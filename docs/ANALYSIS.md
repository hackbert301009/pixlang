# PixLang – Projektanalyse (Stand v0.4 → v0.5)

Diese Analyse wurde mit einem Agententeam erstellt (Sprachkern · Commands & Runtime · Editor)
und dient als Grundlage für den Umbau zu v0.5.

## Ausgangslage

| Bereich | Stand v0.4 |
|---|---|
| Code | ~5 000 Zeilen Python, ~1 000 Zeilen JS/CSS, 35 Builtin-Commands |
| Tests | 287 Tests, **4 rot** (`HISTOGRAM_SAVE` bricht mit OpenCV 5: `calcHist` liefert 1-D) |
| Lint | `ruff check` meldet ~150 Befunde → CI-Lint-Job ist rot |
| CI | läuft nur auf `main`/`develop`, Feature-Branches werden nie geprüft |

## Kritische Befunde

### Sprachkern
- **Parser hängt endlos** bei einem verirrten `END`/`ENDIF` auf Top-Level – über `/api/parse` friert damit auch der Editor-Server ein.
- **INCLUDE-Zyklen** laufen bis `RecursionError`; verschachtelte INCLUDEs lösen Pfade relativ zur Root-Datei statt zur einbindenden Datei auf.
- **CRLF-Dateien** (Windows) lassen sich nicht lexen; negative Floats, `.5`, `1e3`, Hex/Binär fehlen.
- **Zeilenumbrüche beenden keine Statements** – `BLUR 3⏎5` wird zu `BLUR [3, 5]`.
- Keine Ausdrücke: `SET`, `IF`, `ASSERT` akzeptieren nur Literale, kein `ELSE`, kein Rechnen.
- `$ITER` wird von verschachtelten Schleifen überschrieben; Fehler ohne Spalte/Datei.
- Linter ignoriert ROI- und INCLUDE-Inhalte, mehrere False Positives; Großteil der `pixlang.toml`-Optionen wird nie gelesen.

### Commands & Runtime
- **Batch-Modus verarbeitet N-mal die erste Datei** (`batch_override` wird nie gelesen), der „Dry Run“ führt die komplette Pipeline inkl. Speichern aus.
- 17 Commands verwerfen Metadaten (z. B. Konturen) → `FIND_CONTOURS → DILATE → DRAW_BOUNDING_BOXES` zeichnet nichts.
- Keine Typkonvertierung der Argumente (`SET k 5.0` + `BLUR $k` → rohe OpenCV-Fehler), uint16/float/BGRA-Bilder brechen viele Commands.
- Watcher startet nach einem Fehler alle 0,5 s endlos neu; Plugins kollidieren beim Reload.
- **Sicherheit:** lokale `*.plugins.py` und `~/.pixlang/plugins` werden ungefragt ausgeführt.
- Tests schreiben ins Repo (`out/`, `examples/`).

### Editor
- Syntax-Highlighting für Commands defekt (ungültige Token-Namen), Klick auf Command fügt Docstring ein → Syntaxfehler.
- Flowchart zeichnet Block-Inhalte *unter* `ENDIF`, Labels zeigen `VarRef(line=…)`.
- Lint-Meldungen nie sichtbar, Fehler ohne Zeilenmarkierung, Vorschau nach Fehler veraltet ohne Hinweis.
- Kein Light-Theme, kein Responsive-Layout, Emoji-Icons, Kontrast unter WCAG AA.
- **Sicherheit (hoch):** `get_json(force=True)` erlaubt CSRF von beliebigen Webseiten auf `localhost` (Pipeline ausführen, Dateien schreiben); `INCLUDE "/etc/…"` leakt Dateiinhalte; kein Host-Check, kein Timeout.

## Zielbild v0.5

**Sprache – tiefer an die Hardware, aber einfacher:** vier Spezialmechanismen
(CHECKPOINT, ROI, INCLUDE, BareIdent) werden durch drei allgemeine ersetzt –
**Buffer**, **Views/Pointer** und **Module mit Linker** – plus ein echter Ausdrucks-Parser.

```pxl
FROM "lib/filters.pxl" IMPORT SHARPEN_TWICE, clamp

ALLOC canvas 640 480 3 u8 0        # typisierter Speicher
LOAD "sample.jpg"
BUF orig                            # Kopie des aktuellen Bildes (_)
SET p &orig[10:110, 20:220]         # Pointer = Live-View in einen Buffer
WITH *p
    SHARPEN_TWICE 3
END
SET px orig[12, 40, 2]              # PEEK
SET orig[12, 40, 2] 255             # POKE
SET bits (orig & 0xF0) | (orig >> 4)
FREE orig                           # Zugriff über p danach → use-after-free-Fehler
```

- Ausdrücke mit Bit-Operatoren (`& | ^ ~ << >>`), `IF/ELIF/ELSE`, `WHILE`, `FOR … TO … STEP`, `BREAK/CONTINUE`
- `FUNC … ENDFUNC` mit eigenem Scope, `RETURN`, `GLOBAL`
- **Linker:** Objekt-Units pro Datei, Symbolauflösung über Dateigrenzen, Fehler für Zyklen, doppelte Exporte, undefinierte Symbole (`datei:zeile:spalte`), `pixlang link --map`
- Neue hardwarenahe Commands: `CONVERT_DEPTH`, `BITWISE_*`, `SHIFT_*`, `BIT_PLANE`, `SAVE_RAW`/`LOAD_RAW` (rgb565 …), `EXPORT_C_ARRAY` (C-Header für Mikrocontroller), `DITHER`, `STATS` u. v. m.

**Editor:** Neugestaltung mit Design-Tokens (Dark/Light), angedockten, skalierbaren Panels,
echtem Sprach-Support (Autocomplete, Hover-Doku, Lint-Gutter, Fehlermarker), Command-Palette,
Vorher/Nachher-Vergleich, Buffer-Leiste, korrekt verschachteltem Flowchart und gehärtetem Server.

## Nächste Schritte (nach v0.5)
- IR/Bytecode-Stufe mit `pixlang dis` und Step-Debugger im Editor
- ONNX-Inferenz (`INFER`), Tiling für sehr große Bilder
- Kompilierungs-Cache für gelinkte Module
