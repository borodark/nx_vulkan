#!/usr/bin/env bash
# Live bar graph from tegrastats. Written for the Jetson, where nvidia-smi
# reports nothing useful and `clocks.sm` does not exist — GR3D_FREQ is the only
# window onto the DVFS state that has now voided or distorted four measurements
# in this project.
#
#   ssh io@192.168.0.250 tegrastats --interval 1000 | sh scripts/tegrastats_bars.sh
#   tegrastats --interval 1000 | sh scripts/tegrastats_bars.sh     # on the box
#   sh scripts/tegrastats_bars.sh < saved_tegrastats.log           # replay
#
# Every gauge carries a PEAK marker (▏) that never falls. That is the point: a
# clock that boosted once and decayed looks identical to one that never boosted,
# if you only ever see the instantaneous value.
#
# Ctrl-C to stop. Writes a summary to stderr on exit so a piped run still
# leaves you the peaks.

set -u
BAR_CH="${BAR_CH:-█}"
PEAK_CH="${PEAK_CH:-▏}"
ESC=$'\033'
declare -A PEAK=()

cols() { local c; c=$(tput cols 2>/dev/null || echo 80); echo "$((c < 40 ? 40 : c))"; }
WIDTH=$(( $(cols) - 34 )); ((WIDTH < 10)) && WIDTH=10

# colour by fraction of full scale: cool -> hot
hue() {
  local pct=$1
  if   (( pct >= 90 )); then printf '%s[1;31m' "$ESC"
  elif (( pct >= 70 )); then printf '%s[33m'   "$ESC"
  elif (( pct >= 30 )); then printf '%s[32m'   "$ESC"
  else                       printf '%s[36m'   "$ESC"; fi
}

# bar LABEL VALUE MAX SUFFIX
bar() {
  local label=$1 val=$2 max=$3 suffix=$4 key=$1
  [[ -z "$val" || "$val" == "off" ]] && { printf '  %-10s %s[2m%-*s off%s[0m\n' \
      "$label" "$ESC" "$WIDTH" "" "$ESC"; return; }
  (( max <= 0 )) && max=1
  local pct=$(( val * 100 / max ))
  (( pct > 100 )) && pct=100
  local fill=$(( pct * WIDTH / 100 ))
  local prev=${PEAK[$key]:-0}
  (( val > prev )) && PEAK[$key]=$val && prev=$val
  local pmark=$(( prev * 100 / max * WIDTH / 100 ))
  (( pmark >= WIDTH )) && pmark=$((WIDTH - 1))

  local line="" i
  for ((i = 0; i < WIDTH; i++)); do
    if   ((i < fill));    then line+="$BAR_CH"
    elif ((i == pmark));  then line+="$PEAK_CH"
    else                       line+=" "; fi
  done
  printf '  %-10s %s%s%s[0m %s\n' "$label" "$(hue "$pct")" "$line" "$ESC" "$suffix"
}

# tegrastats fields vary by JetPack; every parse below is optional and a missing
# one renders as "off" rather than aborting the display.
render() {
  local line=$1
  printf '%s[H%s[J' "$ESC" "$ESC"
  printf '  %s[1mtegrastats%s[0m  %s   peak marker: %s\n\n' \
    "$ESC" "$ESC" "$(date +%H:%M:%S)" "$PEAK_CH"

  # GR3D_FREQ 45%@921  |  GR3D_FREQ 45%
  local g3 gfreq
  g3=$(sed -n 's/.*GR3D_FREQ \([0-9]\+\)%.*/\1/p' <<<"$line")
  gfreq=$(sed -n 's/.*GR3D_FREQ [0-9]\+%@\([0-9]\+\).*/\1/p' <<<"$line")
  bar "GPU util" "${g3:-}" 100 "${g3:-?}%"
  [[ -n "$gfreq" ]] && bar "GPU clock" "$gfreq" 1000 "${gfreq} MHz"

  local emc efreq
  emc=$(sed -n 's/.*EMC_FREQ \([0-9]\+\)%.*/\1/p' <<<"$line")
  efreq=$(sed -n 's/.*EMC_FREQ [0-9]\+%@\([0-9]\+\).*/\1/p' <<<"$line")
  bar "EMC" "${emc:-}" 100 "${emc:-?}%${efreq:+  ${efreq} MHz}"
  echo

  # CPU [10%@1224,off,off,5%@1224]
  local cpus n=0 c pct freq
  cpus=$(sed -n 's/.*CPU \[\([^]]*\)\].*/\1/p' <<<"$line")
  IFS=',' read -ra CORE <<<"${cpus:-}"
  for c in "${CORE[@]:-}"; do
    if [[ $c == off ]]; then bar "cpu$n" "off" 100 ""
    else
      pct=${c%%\%*}; freq=${c##*@}
      bar "cpu$n" "$pct" 100 "${pct}%  ${freq} MHz"
    fi
    n=$((n + 1))
  done
  echo

  local used tot
  used=$(sed -n 's/.*RAM \([0-9]\+\)\/[0-9]\+MB.*/\1/p' <<<"$line")
  tot=$(sed -n 's/.*RAM [0-9]\+\/\([0-9]\+\)MB.*/\1/p' <<<"$line")
  [[ -n "$used" ]] && bar "RAM" "$used" "${tot:-4096}" "${used}/${tot} MB"

  # temps are decimal; bar takes integers, so truncate for the gauge only
  local gt
  gt=$(sed -n 's/.*GPU@\([0-9]\+\)\(\.[0-9]\+\)\?C.*/\1/p' <<<"$line")
  [[ -n "$gt" ]] && bar "GPU temp" "$gt" 100 "$(sed -n 's/.*\(GPU@[0-9.]\+C\).*/\1/p' <<<"$line")"
}

summary() {
  printf '\n%s[1mpeaks this run%s[0m\n' "$ESC" "$ESC" >&2
  local k
  for k in "${!PEAK[@]}"; do printf '  %-10s %s\n' "$k" "${PEAK[$k]}" >&2; done
  printf '%s[?25h' "$ESC" >&2
}
trap summary EXIT INT TERM
printf '%s[?25l' "$ESC"

if [ -t 0 ]; then
  command -v tegrastats >/dev/null || { echo "no stdin and no tegrastats on PATH" >&2; exit 1; }
  tegrastats --interval 1000 | while IFS= read -r l; do render "$l"; done
else
  while IFS= read -r l; do render "$l"; done
fi
