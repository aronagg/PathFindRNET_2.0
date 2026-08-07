# Homography Calibration Point Provenance

All 101 frozen correspondences were selected manually. Newport, Eastgate and SE38th
reuse point sets from prior calibration assets; 116th/NE12th and NE8th were collected
with the visual point-pair workflow. The CSVs do not document whether each point is a
lane corner, marking, curb feature, or another landmark, so point-level feature type
is recorded as `unknown/not documented` rather than inferred.

| scene | points | point provenance | top-view source | landmark type |
| --- | --- | --- | --- | --- |
| 116th / NE12th | 26 | manual_visual_selection_feature_not_documented | old_data/Google_Maps_Pics/Bellevue_116th_NE12th_google_maps.png | unknown/not documented per point |
| 150th / Newport | 20 | manual_existing_calibration_point_feature_not_documented | old_data/Google_Maps_Pics/Bellevue_150th_Newport_google_maps.png | unknown/not documented per point |
| 150th / Eastgate | 13 | manual_existing_calibration_point_feature_not_documented | old_data/Google_Maps_Pics/Bellevue_150th_Eastgate_google_maps.png | unknown/not documented per point |
| 150th / SE38th | 24 | manual_existing_calibration_point_feature_not_documented | old_data/Google_Maps_Pics/Bellevue_150th_SE38th_google_maps.png | unknown/not documented per point |
| NE8th | 18 | manual_visual_selection_feature_not_documented | old_data/Google_Maps_Pics/Bellevue_NE_NE8th_google_maps.png | unknown/not documented per point |

The five `topview_reference_image.png` files are byte-identical to the named files in
`old_data/Google_Maps_Pics`. Repository reports identify these as Google Maps/top-view
screenshots. Camera reference images are extracted traffic-video frames. Checksums,
normalized coordinates, image dimensions, and inlier flags are published in
`results/homography/calibration_correspondences.csv`.

NE8th's five rejected IDs remain auditable in the original CSV and are not silently
deleted. The frozen correspondence export contains only the 18 points actually used.
