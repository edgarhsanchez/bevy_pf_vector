#!/usr/bin/env bash
# Pins benchmark opponents and reference implementations as submodules under
# benchmarks/vendor/. These are COMPARISON TARGETS for the benchmark suite,
# not dependencies of the engine — bevy_pf_vector must stay pure.
# SHAs captured 2026-08-03. Run from the repo root in Git Bash.
set -euo pipefail

add() { # add <url> <path> <sha>
  git submodule add --force "$1" "benchmarks/vendor/$2" || true
  git -C "benchmarks/vendor/$2" fetch --depth 1 origin "$3"
  git -C "benchmarks/vendor/$2" checkout "$3"
}

# --- benchmark opponents / prior art ---
add https://github.com/linebender/vello.git            vello              cd38479cd26a153198916541a9f28fe87e300f28
add https://github.com/linebender/kurbo.git            kurbo              ca273499e3e48bd2de6f02aa8e99a148984e45f3
add https://github.com/linebender/peniko.git           peniko             e1bb9ef26282a2dd48058a3a7154691548fe4980
add https://github.com/rive-app/rive-bevy.git          rive-bevy          378cef706ea256807580d045e470d9dcaff314cb
add https://github.com/nical/lyon.git                  lyon               8071ec066c610b006e58086fea30cd96d4cef153
add https://github.com/servo/pathfinder.git            pathfinder         6c3c0466f451c5bd2007087728cd168798cd64e8
add https://github.com/james-j-obrien/bevy_vector_shapes.git bevy_vector_shapes 623868d437c416216201e0707704fdc988cc4147

# --- reference only; skia is ~1.5 GB, keep it shallow or skip ---
# add https://github.com/google/skia.git               skia               a08d918ebd6a85b03538040ad80da26dc78e387f

echo "done. review .gitmodules, then commit."
