# Independent-Test Preflight Report

- Status: **PASS**
- Timestamp: `2026-08-06T11:24:34+00:00`
- Git commit: `96ee0c4525d2cca9f01c86a15395fa4491774214`
- Frozen configuration hash: `829c95c4f0a012d08433536b22379afad4d5a79122e52f8fd6ac12817882167c`
- Polygon-reference protocol hash: `b3ed9c7a211d461e21d7b3f39a1c9b34793ddcea5fea8036ebf4f02128377d6a`
- Independent-test rows: **27,393**
- Frozen selected configurations: **30**
- No independent-test feature vector or reference-label row was loaded.
- Reference validation at this stage used file hashes and manifest metadata only.
- No development configuration changed.

## Hash Checks

| artifact                               | expected_sha256                                                  | actual_sha256                                                    | status   |
|:---------------------------------------|:-----------------------------------------------------------------|:-----------------------------------------------------------------|:---------|
| frozen_protocol_file                   | 865513b8be4d59365bd430e5909fbd72b6a9d37f3e8b805917af3a80e66a1337 | 865513b8be4d59365bd430e5909fbd72b6a9d37f3e8b805917af3a80e66a1337 | PASS     |
| runner_config                          | 982699aa24f3c2f33b2e9a8e670fc3b6aeef8992d697d71cfd66f4c7449943f9 | 982699aa24f3c2f33b2e9a8e670fc3b6aeef8992d697d71cfd66f4c7449943f9 | PASS     |
| trajectory_manifest                    | 0b26f59168f0c3996290cc1832ddd423321a5988e27fca93e78d2a02ecde7a03 | 0b26f59168f0c3996290cc1832ddd423321a5988e27fca93e78d2a02ecde7a03 | PASS     |
| evaluation_split                       | 0ec1e76f7024ecbea46b832be56ae07ab2d9b16dc57a99654d0d8390d909774f | 0ec1e76f7024ecbea46b832be56ae07ab2d9b16dc57a99654d0d8390d909774f | PASS     |
| split_config                           | 0f5c09fbe9b28dcea5d613020d947b801d0f1cf6bd7ca71598ac959e658d8e7d | 0f5c09fbe9b28dcea5d613020d947b801d0f1cf6bd7ca71598ac959e658d8e7d | PASS     |
| homography_index                       | d33cee67771be75a8a912911ae3b3d7d184a6df0cfa9cbea09bf5dd3d6ff47c8 | d33cee67771be75a8a912911ae3b3d7d184a6df0cfa9cbea09bf5dd3d6ff47c8 | PASS     |
| target_estimates                       | 57664aa66e2f902853e304c1cd2a85d6d68e16836a7cca42de38aeca322d80f0 | 57664aa66e2f902853e304c1cd2a85d6d68e16836a7cca42de38aeca322d80f0 | PASS     |
| target_candidates                      | b9510562e8cecf1533f75ff5547492e532a6c93499b186d5ba6e484fdb18f935 | b9510562e8cecf1533f75ff5547492e532a6c93499b186d5ba6e484fdb18f935 | PASS     |
| target_region_candidates               | 4ec450ab8909975354a194f726b1e6ae38fb6fd5a0bd7ca6e7aa6f758014b207 | 4ec450ab8909975354a194f726b1e6ae38fb6fd5a0bd7ca6e7aa6f758014b207 | PASS     |
| target_provenance                      | 6fbfd74605a4ebaaba7e0b39287ed9af6d20a26e1d7a48539e26bda096b7dd0b | 6fbfd74605a4ebaaba7e0b39287ed9af6d20a26e1d7a48539e26bda096b7dd0b | PASS     |
| model_selection_candidates             | c9243c0adb3135d327b4bbaaa3944feefe4b891289a51490fc5ad38734418469 | c9243c0adb3135d327b4bbaaa3944feefe4b891289a51490fc5ad38734418469 | PASS     |
| selected_configurations                | e57e82e282dc888f556090e289a8e845cebce2bb653d9aac1dd9a978d9c76d2f | e57e82e282dc888f556090e289a8e845cebce2bb653d9aac1dd9a978d9c76d2f | PASS     |
| model_selection_provenance             | 338d096d557d1d08fb3c1eccf05ecfe4cc13900b74492c8e145d71049cd4d825 | 338d096d557d1d08fb3c1eccf05ecfe4cc13900b74492c8e145d71049cd4d825 | PASS     |
| homography:bellevue_116th_ne12th       | 9afd259bd684c188409527be753af4f8bcb572b7fbcada7e17d184c56c2f730a | 9afd259bd684c188409527be753af4f8bcb572b7fbcada7e17d184c56c2f730a | PASS     |
| homography:bellevue_150th_newport      | 1e3a078de822a5dc946669c4f91cbf75f87dc808232e4e5285eecbba61ee735c | 1e3a078de822a5dc946669c4f91cbf75f87dc808232e4e5285eecbba61ee735c | PASS     |
| homography:bellevue_150th_eastgate     | 99bbf7a9fc5b8cca7bfb91b5055a84ff75b47a2f679a0a196a92b197e273f366 | 99bbf7a9fc5b8cca7bfb91b5055a84ff75b47a2f679a0a196a92b197e273f366 | PASS     |
| homography:bellevue_150th_se38th       | 47da6b23ad76ba6fbb101b94fde50acfacc794af17eadfbe52d4a31d4032b51b | 47da6b23ad76ba6fbb101b94fde50acfacc794af17eadfbe52d4a31d4032b51b | PASS     |
| homography:bellevue_ne8th              | d5b5ccd06d2cd9a1bdf356895347bd66ed4e23d388f860fe195d11a07290ddc6 | d5b5ccd06d2cd9a1bdf356895347bd66ed4e23d388f860fe195d11a07290ddc6 | PASS     |
| source_feature:bellevue_116th_ne12th   | ec4bdb9b8a010a7af0535a24952a31d647ee78d045eb95fffebd8a31f0046e09 | ec4bdb9b8a010a7af0535a24952a31d647ee78d045eb95fffebd8a31f0046e09 | PASS     |
| source_feature:bellevue_150th_newport  | d391fcc142e63089e3cf23841e3bd26be03f9247a06da587dd2ebb59189e516f | d391fcc142e63089e3cf23841e3bd26be03f9247a06da587dd2ebb59189e516f | PASS     |
| source_feature:bellevue_150th_eastgate | 40e5fbda861b3a082a97717d2eed15f841892ef8a58068ee4b2747ff6898ed5e | 40e5fbda861b3a082a97717d2eed15f841892ef8a58068ee4b2747ff6898ed5e | PASS     |
| source_feature:bellevue_150th_se38th   | fed46c5dd2373c007616d0b84d16f09481578d40cb97339a15a03f6697ba491a | fed46c5dd2373c007616d0b84d16f09481578d40cb97339a15a03f6697ba491a | PASS     |
| source_feature:bellevue_ne8th          | 400b59562b64745e464cdee99e8e227c903429b933fa26a3f6616a0539a2b968 | 400b59562b64745e464cdee99e8e227c903429b933fa26a3f6616a0539a2b968 | PASS     |
| polygon_reference_protocol             | 6609f29e6251d477089ad210d00ac2160ed0daea73d14c1a26e5fdfa401ede96 | 6609f29e6251d477089ad210d00ac2160ed0daea73d14c1a26e5fdfa401ede96 | PASS     |
| independent_test_reference_export      | 127a52ff80af8906fdc3bb6a7436a222baedd42d8c542cfb2e234e67e43b4f5f | 127a52ff80af8906fdc3bb6a7436a222baedd42d8c542cfb2e234e67e43b4f5f | PASS     |
