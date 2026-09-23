#!/usr/bin/env bash
# Render the garden photoreal on a Colab GPU with the Google Colab CLI.
#
#   render/colab_render.sh                 # all views, final quality, on a T4
#   GPU=L4 VIEWS=front,evening QUALITY=final render/colab_render.sh
#
# Needs: the CLI (pip install google-colab-cli, Python 3.12+), signed in once
# (render/colab_login.py url / code, or `colab --auth=oauth2 sessions` at a terminal), and
# network access to colab.research.google.com. The VM is always released at the end.
set -euo pipefail
cd "$(dirname "$0")/.."

SESSION=${SESSION:-garden-render}
GPU=${GPU:-T4}
VIEWS=${VIEWS:-front,evening,stream,sand,arm,tree}
QUALITY=${QUALITY:-final}
PY=${PY:-python}
COLAB="colab --auth=oauth2"
OUT=out/renders_colab

$PY experiments/export_render_scene.py                       # the model -> out/render_scene
tar czf out/render_bundle.tgz -C out render_scene -C .. render/garden_cycles.py

$COLAB new -s "$SESSION" --gpu "$GPU"
trap '$COLAB stop -s "$SESSION"' EXIT                       # never leave a billable VM running
$COLAB upload -s "$SESSION" out/render_bundle.tgz /content/render_bundle.tgz
printf 'import os\nos.environ["GARDEN_VIEWS"] = "%s"\nos.environ["GARDEN_QUALITY"] = "%s"\n' "$VIEWS" "$QUALITY" \
    | $COLAB exec -s "$SESSION"
$COLAB exec -s "$SESSION" -f render/colab_job.py

mkdir -p "$OUT"
for view in ${VIEWS//,/ }; do
    $COLAB download -s "$SESSION" "/content/renders/$view.png" "$OUT/$view.png"
done
echo "renders in $OUT"
