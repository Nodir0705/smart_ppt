#!/bin/bash
# Per-commit attribution for parser/refine changes (see tasks/lessons.md 2026-10-20):
#   rebuild default index -> tools.remap_ids (maps, talk scripts, expected, adjudication) -> rebuild map index -> clause_replay,
#   with src/ taken from a clean detached worktree of <commit> (uncommitted edits never leak in) and out/ symlinked
#   (recorded STT streams hold cwd-relative wav paths). remap_ids always runs from the main tree (current tool).
# usage: tools/rebuild_replay.sh <commit> <label> [deck ...]     deck names may contain spaces; default AI_Handout_Checker TapHole
# output: rows on stdout (append to out/bench/attribution.md); remap report at out/bench/<deck>/default/remap_<label>.{json,md}
set -euo pipefail
MAIN=/home/netvision/smartppt; S=${SMARTPPT_SCRATCH:-/tmp/smartppt_attrib}; mkdir -p $S
COMMIT=$1; LABEL=$2; WT=$S/wt_$LABEL; shift 2; DECKS=("$@"); [ ${#DECKS[@]} -eq 0 ] && DECKS=(AI_Handout_Checker TapHole)
cd $MAIN && { [ -d $WT ] && git worktree remove --force $WT; true; } && git worktree add -q --detach $WT $COMMIT
trap 'cd $MAIN && git worktree remove --force $WT' EXIT
ln -s $MAIN/out $WT/out   # asr_streams.json holds cwd-relative wav paths
cd $WT
for deck in "${DECKS[@]}"; do
  export D="$MAIN/out/bench/$deck" P="$MAIN/decks/$deck.pptx" MAPS="$MAIN/out/maps/$deck"
  cp "$D/default/index.json" "$D/default/index.prev.json"
  python3 -c 'import os; from src.build_index import build_index; build_index(os.environ["P"], os.environ["D"] + "/default")' >/dev/null 2>&1
  (cd $MAIN && python3 -m tools.remap_ids "$D/default/index.prev.json" "$D/default/index.json" --out "$D/default/remap_$LABEL.json" \
     --apply "$MAPS" "$D/presenter_script.json" "$D/presenter_script_jumpy.json" "$D/presenter_script_meaning.json" \
     "$D/presenter" "$D/presenter_jumpy" "$D/presenter_meaning" > "$D/default/remap_$LABEL.md" 2>&1) || { echo "remap failed for $deck"; tail -20 "$D/default/remap_$LABEL.md"; }
  python3 -c 'import os; from src.build_index import build_index; build_index(os.environ["P"], os.environ["D"] + "/map", slide_map_dir=os.environ["MAPS"])' >/dev/null 2>&1
  echo "## $LABEL ($COMMIT) — $deck"
  LABEL=$LABEL python3 -c 'import json,os; m=json.load(open(os.environ["D"] + "/default/remap_" + os.environ["LABEL"] + ".json")); print("remap: dropped", len(m["dropped"]), "new", len(m["new"]), "retyped", len(m["retyped"]), "text_changed", len(m.get("text_changed", [])))'
  python3 -m tools.clause_replay --bench $MAIN/out/bench --deck "$deck" --holdout 2>/dev/null | grep -E '^\|'
done
