# SLEAP gerbils schema audit

This audit was executed against the downloaded files with the public
`sleap_io.load_slp(path, open_videos=True, lazy=True)` API.  It is a record of
the observed 0.9.2 objects, not a guessed schema.  Full command output and the
HDF5 inspection are retained under
`/data2/usr_for_deadline/MAT_workspace/upstream_audit/sleap_nn/`.

## Runtime

| item | observed value |
|---|---|
| sleap-io | `0.9.2` |
| train/val/test labels | `sleap_io.model.labels.Labels` |
| lazy frame list | `sleap_io.io.slp_lazy.LazyFrameList` |
| loader call | `sio.load_slp(path, open_videos=True, lazy=True)` |
| first labeled frame | `sleap_io.model.labeled_frame.LabeledFrame`, `frame_idx=1813` (train sample) |
| first instance | `sleap_io.model.instance.Instance`, `numpy().shape=(14, 2)` |
| video frame access | public `labeled_frame.video[frame_idx]` |

## Observed split counts

| split | labeled frames | instances | `labels.videos` slots | frame-index range | embedded images |
|---|---:|---:|---:|---|---|
| train | 340 | 1,249 | 23 | 0–7,363 | yes |
| val | 43 | 159 | 23 | 0–7,363 | yes |
| test | 42 | 153 | 23 | 111–6,845 | yes |

The random split files share source-video slots.  Only 11 of the 23 slots have
selected labels; the adapter retains all 23 slots in `session_inventory.json`
and reports the 11 labeled slots separately.  Source video metadata exposed by
the package is `day001.pkg.slp`; the stable source identity is therefore
`day001.pkg.slp#video<index>`, never the train/val/test filename.

## Tracks and skeleton

The four real track names are retained verbatim in private truth:

`female`, `male`, `pup shaved`, `pup unshaved`.

The source node names are also retained verbatim (SLEAP uses no underscores):

`nose`, `lefteye`, `righteye`, `leftear`, `rightear`, `spine1`, `spine2`,
`spine3`, `spine4`, `spine5`, `tail1`, `tail2`, `tail3`, `tail4`.

MAT's canonical species config uses `left_eye`/`right_eye`/`left_ear`/
`right_ear`; `manifests/skeleton.json` records the explicit source-to-canonical
mapping.  No source name is silently rewritten in the audit or private truth.

## Continuous example tracking file

`example_tracking.slp` contains 2,560 labeled frames and 9,744 instances for
one video named `2020-3-10_daytime_5mins_compressedTalmo@3200-5760.mp4`.  The
provider did not identify it as human ground truth, so MAT uses it only for
format/tracking smoke and never for formal HOTA or pose metrics.

## Privacy boundary

`prepared/sleap_gerbils/manifests/observations.jsonl` contains neutral UIDs,
frame/session references, image references and local tracklet UIDs only.  The
fields `gt_identity`, `gt_keypoints`, and `gt_visibility` occur only in
`private_pose_identity_truth.jsonl` and are not passed to model observation
manifests.
