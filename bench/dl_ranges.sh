#!/bin/bash
# Download one file with N parallel curl ranges.
#
# The write must land at the range's exact byte offset. `dd bs=1M seek=N`
# rounds down to a 1 MiB boundary, so every chunk after the first overwrote the
# wrong region — the file size still matched, so the corruption was invisible
# until CogVideoX rendered a pure black clip. `oflag=seek_bytes` with bs=1 makes
# seek an exact byte count; verified 1000 bytes at offset 999 land at 999.
#
# usage: dl_ranges.sh <url> <out> <size> [workers] [sha256]
#
# The sha256 argument is the digest the hub reports for LFS files; when given,
# a mismatched assembly is deleted rather than left as a plausible-looking file.
set -u
url="$1"; out="$2"; size="$3"; workers="${4:-12}"
chunk=$(( (size + workers - 1) / workers ))
mkdir -p "$(dirname "$out")"
truncate -s "$size" "$out"

pids=()
for ((i=0; i<workers; i++)); do
  s=$((i*chunk)); e=$((s+chunk-1))
  ((e >= size)) && e=$((size-1))
  ((s >= size)) && break
  # A short read from curl must fail loudly rather than leave a hole.
  # bs=1M with oflag=seek_bytes: seek counts bytes, not blocks, so it is exact
  # without the byte-at-a-time cost of bs=1 on multi-GB files.
  ( set -o pipefail
    curl -sL --retry 5 --retry-delay 3 --max-time 7200 -r "$s-$e" "$url" 2>/dev/null \
      | dd of="$out" bs=1M seek="$s" oflag=seek_bytes conv=notrunc status=none ) &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
# Leave nothing behind on failure. `truncate` pre-allocates the full size, so a
# failed chunk leaves a file with the correct length and a hole of zeros inside
# it — a later `[ -f "$target" ]` then treats corruption as "already downloaded".
# This is not hypothetical: it silently poisoned an 11 GB T5 checkpoint and a
# 2.8 GB LoRA, both of which passed a size check and failed only at load time.
if [ "$fail" -ne 0 ]; then rm -f "$out"; echo "CHUNK FAILED $out (removed)"; exit 1; fi

actual=$(stat -c%s "$out")
if [ "$actual" -ne "$size" ]; then
  rm -f "$out"; echo "SIZE MISMATCH $out $actual != $size (removed)"; exit 1
fi

# Optional fifth argument: the sha256 the hub publishes for LFS files. A range
# download can succeed at the transport level and still assemble wrong, so the
# only proof the bytes are right is the digest.
digest="${5:-}"
if [ -n "$digest" ]; then
  actual_sum=$(sha256sum "$out" | cut -d' ' -f1)
  if [ "$actual_sum" != "$digest" ]; then
    rm -f "$out"; echo "SHA256 MISMATCH $out (removed)"; exit 1
  fi
fi
echo "OK $out ($(( size/1048576 )) MB)"
