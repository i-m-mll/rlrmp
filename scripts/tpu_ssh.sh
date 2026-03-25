#!/bin/bash
# Retry SSH to TPU VM with exponential backoff
# Usage: tpu_ssh.sh <tpu-name> <zone> <command>
TPU=$1; ZONE=$2; shift 2; CMD="$*"
for i in 1 2 3; do
  result=$(gcloud compute tpus tpu-vm ssh "$TPU" --project=bboddy --zone="$ZONE" --command="$CMD" 2>&1)
  if [ $? -eq 0 ]; then echo "$result"; exit 0; fi
  echo "SSH attempt $i failed, retrying in $((i*5))s..." >&2
  sleep $((i*5))
done
echo "SSH failed after 3 attempts" >&2; echo "$result"; exit 1
