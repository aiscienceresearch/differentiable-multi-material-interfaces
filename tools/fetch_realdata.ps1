# Fetch the CT and produce the probability volume used by src/realdata*.py.
#
# The CT is the example volume shipped with TotalSegmentator's test suite: a real
# thoracoabdominal scan resampled to 3 mm. The segmentation task is
# liver_segments, which is openly licensed and, unlike the default `total` task,
# is a single model, which is what --save_probabilities requires.
#
# TotalSegmentator is installed against tools/seg-constraints.txt so that pip
# does not replace a working CUDA torch build with a generic wheel.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path data | Out-Null

pip install --quiet -c tools/seg-constraints.txt totalsegmentator nibabel vtk

$scripts = Join-Path $env:LOCALAPPDATA `
    "Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts"
if (Test-Path $scripts) { $env:PATH = "$env:PATH;$scripts" }

$ct = "data/example_ct.nii.gz"
if (-not (Test-Path $ct)) {
    curl.exe -sL -o $ct `
        "https://github.com/wasserth/TotalSegmentator/raw/master/tests/reference_files/example_ct.nii.gz"
}

TotalSegmentator -i $ct -o data/seg_liver -ta liver_segments `
    --save_probabilities data/probs_liver.npz

python src/realdata_experiment.py
python src/figures_realdata.py
