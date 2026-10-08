$ErrorActionPreference='Stop'
$base=(Resolve-Path -LiteralPath 'openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/parallel_source_staging_and_mechanism_design_attempt01').Path
$index=Join-Path $base 'cross_tfg_source_proof_pack_index_attempt03.json'
if((Get-Item -LiteralPath $index).Length -ne125723 -or (Get-FileHash -LiteralPath $index).Hash.ToLowerInvariant() -ne'78f2071dc059d439b9ed4fb5a6cf1382c6a6536975d15181988c948f4cccfe24'){throw 'Index copy mismatch'}
$j=Get-Content -LiteralPath $index -Raw|ConvertFrom-Json
$copies=0;$refs=0;$bytes=0L
foreach($e in $j.entries){
 $src=Get-Item -LiteralPath $e.absolute_source_path
 if($src.LinkType -or $src.Length -ne$e.bytes -or (Get-FileHash -LiteralPath $src.FullName).Hash.ToLowerInvariant() -ne$e.sha256){throw 'Source mismatch'}
 if($e.copy){
  if([IO.Path]::IsPathRooted($e.relative_path) -or $e.relative_path -match '(^|[\\/])\.\.([\\/]|$)'){throw 'Unsafe relative path'}
  $dest=[IO.Path]::GetFullPath((Join-Path $base $e.relative_path))
  if(-not $dest.StartsWith($base+[IO.Path]::DirectorySeparatorChar)){throw 'Outside proof'}
  $f=Get-Item -LiteralPath $dest
  if($f.LinkType -or $f.Length -ne$e.bytes -or (Get-FileHash -LiteralPath $dest).Hash.ToLowerInvariant() -ne$e.sha256){throw 'Destination mismatch'}
  $copies++;$bytes+=$f.Length
 }else{$refs++}
}
if($copies -ne180 -or $refs -ne8 -or $bytes -ne1920879){throw 'Proof denominator mismatch'}
$proof=Join-Path $base 'root_actual_proof_copy_qc_attempt01.json'
if(Test-Path -LiteralPath $proof){throw 'QC receipt already exists'}
$r=@{status='PASS_ALL188_SOURCES_AND180_DESTINATION_BYTE_PINS';checked_at_utc=[DateTimeOffset]::UtcNow.ToString('o');source_entries=188;copied_entries=$copies;copied_bytes=$bytes;external_references=$refs;index_bytes=125723;index_sha256='78f2071dc059d439b9ed4fb5a6cf1382c6a6536975d15181988c948f4cccfe24';candidate_scientific_admission=$false;future_dev_executed=$false;direct_d0_training_executed=$false}
$raw=[Text.UTF8Encoding]::new($false).GetBytes(($r|ConvertTo-Json -Depth 5)+[char]10)
$s=[IO.File]::Open($proof,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try{$s.Write($raw);$s.Flush($true)}finally{$s.Dispose()}
$r|ConvertTo-Json -Depth 5
