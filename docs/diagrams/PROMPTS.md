# Diagram prompts

The diagrams were generated with gpt-image from the briefs below; every label was checked by eye against the brief. The dark variants are colour-only edits of the light ones. The example item is synthetic: no SESGO prompt is reproduced.

## item-anatomy

```text
Use: Explanatory figure in the README of an open-source AI-evaluation research repository on GitHub. Readers are researchers; it must read like a figure from a well-designed methods paper, not marketing.
Subject: How one item of a social-bias benchmark works. The same question is asked under two contexts, there are three answer options, and the scoring separates accuracy from direction of error.
Composition: Landscape 16:9, strict 12-column grid, generous margins, three horizontal bands read top to bottom, all elements aligned to the same left edge.
Band 1 (top): two equal cards side by side. Left card header "AMBIGUOUS CONTEXT", body text "Ana and Beto both applied for the same job." and below it in grey "Nobody is named." Right card header "DISAMBIGUATED CONTEXT", body text "Ana and Beto both applied for the same job." and below it in indigo bold "Beto's application had no certificate." Centered under both cards, one wide pill with the text "Who was not qualified for the job?"
Band 2 (middle): three equal option tiles in a row. Tile 1 label "target" with text "Ana, group A". Tile 2 label "other" with text "Beto, group B". Tile 3 label "unknown" with text "Not enough information". A thin arrow from the left card down to tile 3 labelled "correct when ambiguous". A thin arrow from the right card down to tile 2 labelled "correct when disambiguated". Arrows are straight or single right-angle, with small solid arrowheads, never crossing text.
Band 3 (bottom): on the left, three short definition lines: "accuracy: share of correct answers", "Ft: errors toward the target", "Fo: errors toward the other". On the right, a small right triangle: horizontal leg labelled "1 - accuracy", short vertical leg labelled "Ft - Fo", hypotenuse in indigo labelled "bias score". The horizontal leg is about four times longer than the vertical leg.
Style: Flat vector information design in the Swiss tradition. Off-white background #FAFAF7, charcoal #1F2328 lines and text, one muted indigo accent #4B4FA6, light grey card fills. Thin 1.5 px strokes, rounded 6 px corners. Clean grotesque sans-serif typography (like Inter or Helvetica), small uppercase letterspaced headers, monospace for the labels target, other, unknown, accuracy, Ft, Fo. No gradients, no shadows, no glow, no 3D, no people, faces, robots, brains, flags, circuitry, icons or decorative particles.
Text: Render exactly the quoted strings above, spelled exactly, each once, and no other text anywhere. No title, no watermark, no logo.
```

Follow-up edit: the connector arrows were replaced by the two pills inside the `other` and `unknown` tiles.

## pipeline

```text
Use: Pipeline figure in the README of an open-source AI-evaluation research repository on GitHub. It must read like a precise systems figure from a methods paper.
Subject: A left-to-right evaluation pipeline with seven stages and two controls attached to the run stage.
Composition: Landscape 16:9, strict grid, generous margins. One horizontal row of seven equal rounded boxes connected by short straight arrows with small solid arrowheads, left to right, vertically centered slightly above the middle. Each box has a bold one-line title and a smaller monospace second line. Boxes in order:
1. title "Pinned commit", second line "mvrobles/SESGO"
2. title "Local conversion", second line "make data"
3. title "Inspect task", second line "sesgo/sesgo"
4. title "Provider run", second line "sesgo-run"
5. title "Resumable logs", second line "logs/*.eval"
6. title "Aggregates", second line "history.jsonl"
7. title "Report + figures", second line "REPORT.md"
Boxes 1, 2 and 5 have a dashed outline. Boxes 3, 4, 6 and 7 have a solid outline with an indigo left edge.
Below box 4, two control boxes with indigo outlines, side by side, each connected to box 4 by a thin vertical line: left control title "Budget gate", second line "spent + estimate > budget"; right control title "Spend ledger", second line "ledger.jsonl".
Bottom-left legend, two small swatches: a dashed swatch with the text "git-ignored, never published" and a solid indigo-edged swatch with the text "committed".
Style: Flat vector information design in the Swiss tradition. Off-white background #FAFAF7, charcoal #1F2328 lines and text, one muted indigo accent #4B4FA6, very light grey box fills. Thin 1.5 px strokes, 6 px rounded corners, clean grotesque sans-serif (like Inter), monospace second lines. No gradients, no shadows, no glow, no 3D, no icons, no people, no decorative elements.
Text: Render exactly the quoted strings above, spelled exactly, each once, and no other text anywhere. No title, no watermark, no numbering.
```

Cropped vertically after generation.
