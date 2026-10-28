# Corpus provenance

Source: scrollmapper/bible_databases, `formats/json/KJV.json`.

https://github.com/scrollmapper/bible_databases/blob/master/formats/json/KJV.json

Translation: conventional 66-book King James Version (1769), no Apocrypha. The source describes its KJV content as public-domain text. This project extracts verse text only; it does not include the upstream database software, annotations, or morphology fields. All text in train.bin and val.bin is UTF-8; extensions are binary because values are used directly as token IDs.

The original source digest, exact extraction method, split chapter IDs, and output checksums are in manifest.json. The streams contain 31,102 verses across 1,189 chapters before splitting. A deterministic hash ranking selects 59 validation chapters and 1,130 training chapters. These files contain no pretrained model data or generated examples.

The project tests data identity and canon size. It has not manually compared every word against a printed KJV edition.
