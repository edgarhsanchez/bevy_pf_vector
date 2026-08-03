#!/usr/bin/env bash
# Adds the upstream renderers as pinned submodules under vendor/.
# SHAs captured 2026-08-03. Run from the root of the target repo.
set -euo pipefail

add() { # add <url> <path> <sha>
  git submodule add --force "$1" "vendor/$2" || true
  git -C "vendor/$2" fetch --depth 1 origin "$3"
  git -C "vendor/$2" checkout "$3"
}

# --- primary: the renderer we bind against ---
add https://github.com/rive-app/rive-runtime.git       rive-runtime       4ac7b32798da0482e441ef09304dc3b480ed3ee5
add https://github.com/rive-app/rive-rs.git            rive-rs            3d26aa1c3865260e5ef9751ad0172dcc197ff8eb
add https://github.com/rive-app/rive-bevy.git          rive-bevy          378cef706ea256807580d045e470d9dcaff314cb

# --- comparison / prior art ---
add https://github.com/linebender/vello.git            vello              cd38479cd26a153198916541a9f28fe87e300f28
add https://github.com/linebender/kurbo.git            kurbo              ca273499e3e48bd2de6f02aa8e99a148984e45f3
add https://github.com/linebender/peniko.git           peniko             e1bb9ef26282a2dd48058a3a7154691548fe4980
add https://github.com/linebender/xilem.git            xilem              78d4f4b6878a1c2e2884edd72d7ecd83f1f6f704
add https://github.com/nical/lyon.git                  lyon               8071ec066c610b006e58086fea30cd96d4cef153
add https://github.com/servo/pathfinder.git            pathfinder         6c3c0466f451c5bd2007087728cd168798cd64e8
add https://github.com/james-j-obrien/bevy_vector_shapes.git bevy_vector_shapes 623868d437c416216201e0707704fdc988cc4147

# --- reference only; skia is ~1.5 GB, keep it shallow or skip ---
# add https://github.com/google/skia.git               skia               a08d918ebd6a85b03538040ad80da26dc78e387f

git submodule update --init --recursive vendor/rive-runtime
echo "done. review .gitmodules, then commit."
