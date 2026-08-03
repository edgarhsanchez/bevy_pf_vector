# Creates E:\github\bevy_pf_vector, initializes git, and pushes it public.
# Requires: git, gh (authenticated via `gh auth login`).
# Run from the folder where you extracted this bundle.

$ErrorActionPreference = "Stop"

$Target = "E:\github\bevy_pf_vector"
$Src    = $PSScriptRoot

if (Test-Path $Target) {
    throw "$Target already exists. Move or remove it first."
}

Write-Host "Creating $Target ..."
New-Item -ItemType Directory -Path $Target -Force | Out-Null

Get-ChildItem -Path $Src -Exclude "init-and-push.ps1" |
    Copy-Item -Destination $Target -Recurse -Force

Set-Location $Target

git init -b main
git add -A
git commit -m "scaffold: bevy_pf_vector renderer design, backend seam, vendor pins"

# Creates the repo under your account, sets origin, pushes main.
gh repo create bevy_pf_vector --public --source=. --remote=origin --push

Write-Host ""
Write-Host "Done. Repo URL:"
gh repo view --json url --jq .url
