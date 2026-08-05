---
name: nas-access
description: Use when accessing the shared Synology NAS, HDTF data, or related files from WSL/Linux, especially with account dhg_mm and DSM/FileStation or SSHFS.
---

# NAS Access

## Fixed endpoints

- NAS host: `10.108.21.182`
- NAS account: `dhg_mm`
- DSM HTTPS: `https://10.108.21.182:5001`
- FileStation share: `/dhg_mm`
- HDTF data: `/dhg_mm/HDTF`
- Existing 114 mount: `/data2/zql/pub_nas` on `10.1.114.114:1986`
- Project NAS notes: `docs/deployment/nas1.jpg`, `nas2.jpg`, `nas3.jpg`

## Credential rule

The password must be supplied at runtime through `NAS_DHG_MM_PASSWORD`. Never put it in `SKILL.md`, `.env`, shell command arguments, logs, or tracked files.

For an interactive shell, avoid shell history:

```bash
read -rsp 'NAS password: ' NAS_DHG_MM_PASSWORD
printf '\n'
export NAS_DHG_MM_PASSWORD
```

Disable shell tracing before using the variable: `set +x`.
Unset it after use with `unset NAS_DHG_MM_PASSWORD`.

## Preferred access: DSM/FileStation

DSM login is known to work even when SSH password login is rejected. Use `curl --data-urlencode 'passwd@-'` so the password is read from stdin rather than placed in a process argument:

```bash
: "${NAS_DHG_MM_PASSWORD:?Set NAS_DHG_MM_PASSWORD in the current shell}"
set +x 2>/dev/null || true

auth_json="$({
  printf '%s' "$NAS_DHG_MM_PASSWORD"
} | curl -k -sS --max-time 15 -X POST \
  'https://10.108.21.182:5001/webapi/auth.cgi' \
  --data-urlencode 'api=SYNO.API.Auth' \
  --data-urlencode 'version=6' \
  --data-urlencode 'method=login' \
  --data-urlencode 'session=FileStation' \
  --data-urlencode 'format=sid' \
  --data-urlencode 'enable_syno_token=yes' \
  --data-urlencode 'account=dhg_mm' \
  --data-urlencode 'passwd@-')"

sid="$(printf '%s' "$auth_json" | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["data"]["sid"])')"
token="$(printf '%s' "$auth_json" | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["data"]["synotoken"])')"

curl -k -sS -G 'https://10.108.21.182:5001/webapi/entry.cgi' \
  --data-urlencode 'api=SYNO.FileStation.List' \
  --data-urlencode 'version=1' \
  --data-urlencode 'method=list_share' \
  --data-urlencode "SynoToken=$token" \
  --data-urlencode "_sid=$sid"
```

List HDTF with `version=2`, `method=list`, and `folder_path=/dhg_mm/HDTF`. Do not assume `/`, `/home/dhg_mm`, or `/volume1/HDTF` has the same meaning in SSHFS and FileStation; verify the path through FileStation first.

## Optional SSHFS mount

Use SSHFS only after SSH authentication and the remote path have been independently confirmed. The deployment images show `/home/<username>`, but the verified FileStation path is `/dhg_mm`; these are not interchangeable without NAS configuration confirmation.

```bash
command -v sshfs || { printf '%s\n' 'sshfs is required'; exit 1; }
: "${NAS_SSH_REMOTE_PATH:?Set the confirmed SSHFS remote path, for example /home/dhg_mm}"
MOUNT_POINT=/tmp/tts-exp-nas
if mountpoint -q "$MOUNT_POINT"; then
  if ls -A "$MOUNT_POINT" >/dev/null 2>&1; then
    printf 'NAS already mounted at %s\n' "$MOUNT_POINT"
    exit 0
  fi
  fusermount3 -uz "$MOUNT_POINT" 2>/dev/null || fusermount -uz "$MOUNT_POINT"
fi

mkdir -p "$MOUNT_POINT"
set +x 2>/dev/null || true

printf '%s\n' "$NAS_DHG_MM_PASSWORD" | sshfs \
  -o password_stdin,StrictHostKeyChecking=accept-new,uid="$(id -u)",gid="$(id -g)",umask=022,reconnect \
  "dhg_mm@10.108.21.182:${NAS_SSH_REMOTE_PATH}" "$MOUNT_POINT"
mountpoint -q "$MOUNT_POINT" || { printf '%s\n' 'NAS mount is not active' >&2; exit 1; }
test -d "$MOUNT_POINT/HDTF" || { printf '%s\n' 'HDTF path not found' >&2; exit 1; }
```

Unmount with `fusermount3 -u "$MOUNT_POINT"` (or `fusermount -u`). Use `-z` only for a stale `Transport endpoint is not connected` mount, and verify the mount point before running the pipeline.

## Important distinction

The `wjj` account logs into the 114 server. The `dhg_mm` account belongs to the NAS. The persistent 114 SSHFS mount is owned by `zql`, so `wjj` may see `Permission denied` even when the NAS credential is valid.
