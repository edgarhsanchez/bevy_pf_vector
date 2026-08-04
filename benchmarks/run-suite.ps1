# The standard benchmark suite.
#
# Every performance claim in this repo must come from here. Tiers run from the
# HUD sizes bevy_pf actually ships to a one-million-element top end, because
# the interesting behaviour of a tessellate-once-and-instance renderer is how
# it SCALES: fixed per-frame overhead dominates at small counts and vanishes
# at large ones, so a single element count can be made to prove either
# conclusion. Measuring the curve is the point.
#
#   ./benchmarks/run-suite.ps1                     # engine only, all tiers
#   ./benchmarks/run-suite.ps1 -Backends engine,shapes,vello
#   ./benchmarks/run-suite.ps1 -Tiers 1000000      # just the top end
#
# Results land in benchmarks/results/ as JSON, one file per run.

param(
    [string[]] $Backends = @("engine"),
    # 1_000_000 is the standard top end. It is far past any real HUD, and that
    # is deliberate: it is where per-element CPU work (extract, sort, batch)
    # separates from per-frame GPU work, which is what the design is betting on.
    [int[]] $Tiers = @(200, 1000, 5000, 50000, 1000000),
    [int] $Frames = 600,
    [int] $Warmup = 120
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

foreach ($backend in $Backends) {
    foreach ($tier in $Tiers) {
        # The heaviest tiers spend real time just spawning entities; shorten
        # the sample so a full sweep stays practical without changing what is
        # measured (percentiles over a settled window).
        $frames = if ($tier -ge 500000) { [Math]::Min($Frames, 180) } else { $Frames }
        $warmup = if ($tier -ge 500000) { [Math]::Min($Warmup, 40) } else { $Warmup }

        Write-Host "== $backend / $tier elements ==" -ForegroundColor Cyan
        Push-Location $root
        try {
            & cargo run --release -p benchmarks -- `
                --backend $backend --elements $tier `
                --frames $frames --warmup $warmup 2>&1 |
                Select-String "^cpu_frame_ms|^render/vector_pass/elapsed_gpu|^=="
        } finally {
            Pop-Location
        }
    }
}
