# Gerbil persistent-ID audit (provider-label scope)

Status: `VERIFIED` for the data sanity audit; this is not an identity-model
result.  The contact sheet is private workspace output at
`MAT_workspace/runs/data_sanity/gerbils_gt_contact_sheet.png` (seed 17,
eight randomly selected train frames).

The prepared SLEAP manifests contain 23 source-video slots and 11 slots with
labeled frames.  The provider track names visible in the private truth are
`female`, `male`, `pup shaved`, and `pup unshaved`; every one of the 11 labeled
sessions contains all four names in this manifest.  This is evidence that the
provider labels are present across sessions, not a MAT-generated cross-day
mapping.  Shaved versus unshaved appearance is a known nuisance and the names
are therefore reported only as `true_for_provider_labeled_sessions`.

`observations.jsonl` remains neutral and contains no identity, keypoint, or
visibility truth.  The contact sheet and this audit read the private truth
only for data-quality inspection.  S0 enrollment and query evaluation must
still use a fixed reference mapping and an evaluator-only Hungarian assignment;
no predicted identity is being presented as ground truth here.

The current pose smoke checkpoint has two optimizer steps and is not used for
this audit or for formal full-run resolution.  Formal pose and B0 identity
metrics remain `null` until a full checkpoint produces non-zero predictions.
