"""The family's publicly scored artifacts: the only ground truth we have about
the hidden test set.

Each row is (path, public score or None, account, identity basis). Public scores
were read from https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/
(2026-09-25/26) and cross-checked against the family's own anchor ledger
(5GEMSDOE `data/evidence/leaderboard_anchor/leaderboard_anchor.json`) and the
submission notes quoted by the account owners. Identity bases:

  CERTAIN               the account owner's note names the file, or the site
                        offers exactly one file and its sha256 matches
  CERTAIN-BY-EXCLUSION  the only submission artifact that repo/site publishes
  PROBABLE - FLAGGED    inferred from a pre-registered upload order; the score
                        differs from the byte-identical sibling, so it cannot be
                        that file, but the DrivenData submissions page was never
                        read (needs login). Do not treat as verified.

Paths point at sparse checkouts of the sibling repositories under /tmp/sib; see
`scripts/audit_siblings.py` for the clone procedure.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIB = Path("/tmp/sib")

ANCHORS: list[tuple[str, float | None, str | None, str]] = [

    ("/tmp/sib/5GEMSDOE/data/evidence/leaderboard_anchor/gemsdoe-ens12-adopted-7f00890a.tif",
     0.1563, "extradr19", "CERTAIN (only file offered by GEMSDOE site; sha matches committed artifact)"),
    ("/tmp/sib/5GEMSDOE/data/evidence/leaderboard_anchor/gemsdoe2-dual-family-union-f68e590f.tif",
     0.1560, "smashi34", "PROBABLE (GEMSDOE2 pre-registered union-first; 0.1560 != 0.1563 so it is not the byte-identical recall arm) - FLAGGED"),
    ("/tmp/sib/5GEMSDOE/data/evidence/leaderboard_anchor/pindrop-v4-nodes-f347b70daa.tif",
     0.1193, "smrtdoog5", "CERTAIN (owner note names f347b70daa 'SUBMIT FIRST Pindrop nodes')"),
    ("/tmp/sib/5GEMSDOE/data/evidence/leaderboard_anchor/pindrop-v4-ridge-4e03fc9705.tif",
     0.1152, "SDCF9", "CERTAIN (owner note names 4e03fc9705 'CONTROL dense ridge')"),
    ("/tmp/sib/5GEMSDOE/data/evidence/leaderboard_anchor/pindrop-v4-discovery-37f9d5b855.tif",
     0.0830, "wbg1", "CERTAIN (owner note names 37f9d5b855 'SUBMIT SECOND catalogue-gap target')"),
    ("/tmp/sib/6GEMSDOE/downloads/gems6_hgb88-topk03_33cec71ff0.tif",
     0.0286, "(6GEMSDOE site account)", "CERTAIN-BY-EXCLUSION (only .tif the 6GEMSDOE repo/site offers)"),
    ("/tmp/sib/GEMSDOE4/data/evidence/combined/submission.tif",
     0.0343, "(GEMSDOE4 site account)", "CERTAIN-BY-EXCLUSION (only submission artifact committed by GEMSDOE4)"),
    ("/tmp/sib/5GEMSDOE/data/evidence/leaderboard_anchor/gemsdoe2-precision-arm-8bce5dfe.tif",
     None, None, "unscored arm"),
    ("/tmp/sib/5GEMSDOE/data/evidence/leaderboard_anchor/gemsdoe2-extension-arm-ad5ba911.tif",
     None, None, "unscored arm"),
    (str(REPO / "downloads/submission.tif"), None, None,
     "8GEMSDOE apex v1 (built, never uploaded - no public score exists)"),
]


