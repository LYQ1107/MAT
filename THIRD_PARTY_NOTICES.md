# Third-party notices

This repository contains no third-party source or model binary. The following are
references/locked dependencies to be installed only after a separate license and
offline-asset audit:

- DeepLabCut source: upstream project license at the locked commit; SuperAnimal
  weights/config have separate model terms and are not included here.
- FoundationVision ByteTrack: upstream license and the minimal detection-index
  patch in `patches/bytetrack_detection_index.patch`; no YOLOX training tree is vendored.
- WildlifeTools: MIT source license at the locked commit. MegaDescriptor-T-224
  weights are separate `CC-BY-NC-4.0` assets and are not redistributed here.
- idtracker.ai, TrackEval and CowIDentifier: use their upstream licenses only in
  their own isolated environments; no source is copied into MAT.
- Rat ID, PigReID, PigTracking and MultiCamCows2024: provider data licenses must
  be recorded in the asset receipt before import; no data or derived crops are in Git.

