# Fetch FlexiCubes at the commit the comparison in the paper was run against.
#
# Only `flexicubes.py` and `tables.py` are needed. They import nothing but
# torch, so none of the rendering dependencies in their examples (nvdiffrast,
# kaolin) have to be installed.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$dest = Join-Path $here "FlexiCubes"
$commit = "4cc7d6c3d0cee83c011ce36721b81adff0dd7db6"

if (Test-Path $dest) {
    Write-Host "already present at $dest"
    exit 0
}

git clone https://github.com/nv-tlabs/FlexiCubes.git $dest
git -C $dest checkout --quiet $commit
Write-Host "FlexiCubes at $commit in $dest"
