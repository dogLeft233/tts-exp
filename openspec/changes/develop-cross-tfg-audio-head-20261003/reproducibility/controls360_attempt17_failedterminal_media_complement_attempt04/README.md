# Controls360 attempt17 failed-terminal media archive proof

This folder contains small byte-pinned receipts and independent QC records for the actual excluded-media complement archive. It intentionally does not contain the 653,472,590-byte TAR or the 6,430,409-byte compact source index.

The actual controls360 denominator was 360 planned rows: 120 PASS and 240 FAILED; run status was COMPLETE_WITH_FAILURES and the result manifest status was INCOMPLETE. The archive contains 26,328 excluded media files (682,851,546 bytes) in the 26,331-member TAR.

The TAR was copied to the mounted B: CloudFS/CloudDrive volume and reread there byte-for-byte. cloud_sync_confirmed is false; this records the mounted-volume copy only. Compact snapshot blobs remain separately required for full reconstruction. Media were not extracted or deleted, and no model or scoring execution was performed for this archival step.

proof_copy_manifest.json records original source path, byte count, raw SHA-256, and destination byte count/hash for each exact-byte copy.
