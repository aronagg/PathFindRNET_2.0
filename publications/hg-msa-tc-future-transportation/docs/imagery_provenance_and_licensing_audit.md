# Imagery Provenance and Licensing Audit

## Evidence found

- Each `topview_reference_image.png` is byte-identical to a scene file under
  `old_data/Google_Maps_Pics`; repository reports describe the files as Google Maps
  screenshots/top-view images.
- The five camera reference images are extracted Traffic Node Video Dataset frames.
- No acquisition date, zoom level, map coordinates, Google attribution metadata, or
  redistribution permission is stored alongside the top-view screenshots.
- Point coordinates and matrices are reproducible, but they do not resolve image-use
  rights.

## Publication handling

This audit does not make a legal conclusion. Because redistribution permission and
attribution metadata are not documented, the Google-derived raster images should not
be included in the Task 08 review ZIP or redistributed manuscript supplement. Use
camera frames, blank-coordinate reprojection plots, author-created schematic diagrams,
or an appropriately attributed OpenStreetMap-derived figure where licensing permits.
The Task 08 figures deliberately plot destination correspondences on a blank coordinate
canvas and mark every generated figure as free of embedded Google imagery.

Traffic-video frames remain subject to the Traffic Node Video Dataset distribution
conditions and should be handled under the dataset's terms. Exact source-image hashes
are retained locally for reproducibility without copying the source rasters into the
review ZIP.
