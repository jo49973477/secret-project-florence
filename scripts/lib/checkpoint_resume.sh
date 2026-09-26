#!/usr/bin/env bash

# Shared checkpoint/resume setup for GR00T training launchers.
# The caller must set OUTPUT_DIR before invoking checkpoint_resume_configure.

checkpoint_resume_is_resumable() {
    local checkpoint="$1"
    local base="${checkpoint##*/}"
    local optimizer_state=""
    local scheduler_state=""
    local model_state=""
    local rng_state=""

    [[ -d "${checkpoint}" ]] || return 1
    [[ "${base}" =~ ^checkpoint-[0-9]+$ ]] || return 1
    [[ -f "${checkpoint}/trainer_state.json" ]] || return 1

    if [[ -f "${checkpoint}/optimizer.pt" ]]; then
        optimizer_state="${checkpoint}/optimizer.pt"
    else
        optimizer_state="$(find "${checkpoint}" -type f -name '*optim_states.pt' -print -quit)"
    fi
    if [[ -f "${checkpoint}/scheduler.pt" ]]; then
        scheduler_state="${checkpoint}/scheduler.pt"
    else
        scheduler_state="$(find "${checkpoint}" -type f -name '*model_states.pt' -print -quit)"
    fi
    rng_state="$(find "${checkpoint}" -maxdepth 1 -type f -name 'rng_state*.pth' -print -quit)"
    model_state="$(find "${checkpoint}" -type f \( -name '*model_states.pt' -o -name 'pytorch_model*.bin' -o -name '*.safetensors' \) -print -quit)"

    [[ -n "${optimizer_state}" && -n "${scheduler_state}" && -n "${rng_state}" && -n "${model_state}" ]]
}

checkpoint_resume_find_latest() {
    local output_dir="$1"
    local candidate base step
    local latest=""
    local latest_step=-1

    shopt -s nullglob
    for candidate in "${output_dir}"/checkpoint-*; do
        [[ -d "${candidate}" ]] || continue
        base="${candidate##*/}"
        step="${base#checkpoint-}"
        [[ "${step}" =~ ^[0-9]+$ ]] || continue
        checkpoint_resume_is_resumable "${candidate}" || continue
        if (( 10#${step} > latest_step )); then
            latest_step=$((10#${step}))
            latest="${candidate}"
        fi
    done
    shopt -u nullglob

    printf '%s\n' "${latest}"
}

checkpoint_resume_validate_target() {
    local checkpoint="$1"
    local base="${checkpoint##*/}"
    local optimizer_state=""
    local scheduler_state=""
    local model_state=""
    local rng_state=""
    local -a missing=()

    if [[ ! -d "${checkpoint}" ]] || [[ ! "${base}" =~ ^checkpoint-[0-9]+$ ]]; then
        echo "[ERROR] Invalid checkpoint directory: ${checkpoint}" >&2
        exit 1
    fi
    if [[ ! -f "${checkpoint}/trainer_state.json" ]]; then
        missing+=("trainer state")
    fi

    if [[ -f "${checkpoint}/optimizer.pt" ]]; then
        optimizer_state="${checkpoint}/optimizer.pt"
    else
        optimizer_state="$(find "${checkpoint}" -type f -name '*optim_states.pt' -print -quit)"
    fi
    if [[ -f "${checkpoint}/scheduler.pt" ]]; then
        scheduler_state="${checkpoint}/scheduler.pt"
    else
        scheduler_state="$(find "${checkpoint}" -type f -name '*model_states.pt' -print -quit)"
    fi
    rng_state="$(find "${checkpoint}" -maxdepth 1 -type f -name 'rng_state*.pth' -print -quit)"
    model_state="$(find "${checkpoint}" -type f \( -name '*model_states.pt' -o -name 'pytorch_model*.bin' -o -name '*.safetensors' \) -print -quit)"

    [[ -n "${optimizer_state}" ]] || missing+=("optimizer state")
    [[ -n "${scheduler_state}" ]] || missing+=("scheduler state")
    [[ -n "${model_state}" ]] || missing+=("model partitions")
    [[ -n "${rng_state}" ]] || missing+=("RNG state")

    if (( ${#missing[@]} > 0 )); then
        echo "[ERROR] Checkpoint is not resumable; missing ${missing[*]}: ${checkpoint}" >&2
        exit 1
    fi
}

checkpoint_resume_configure() {
    local output_dir="$1"
    local default_save_total_limit="${2:-5}"
    local latest_checkpoint=""

    SAVE_STEPS="${SAVE_STEPS:-100}"
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-${default_save_total_limit}}"
    RESUME="${RESUME:-0}"
    RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
    SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-0}"

    if [[ ! "${SAVE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] SAVE_STEPS must be a positive integer, got: ${SAVE_STEPS}" >&2
        exit 1
    fi
    if [[ ! "${SAVE_TOTAL_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] SAVE_TOTAL_LIMIT must be a positive integer, got: ${SAVE_TOTAL_LIMIT}" >&2
        exit 1
    fi
    if [[ ! "${RESUME}" =~ ^[01]$ ]]; then
        echo "[ERROR] RESUME must be 0 or 1, got: ${RESUME}" >&2
        exit 1
    fi
    if [[ ! "${SAVE_ONLY_MODEL}" =~ ^[01]$ ]]; then
        echo "[ERROR] SAVE_ONLY_MODEL must be 0 or 1, got: ${SAVE_ONLY_MODEL}" >&2
        exit 1
    fi

    # An explicit checkpoint path always requests resume, even when RESUME was omitted.
    if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
        RESUME=1
    fi

    if [[ "${RESUME}" == "1" && "${SAVE_ONLY_MODEL}" == "1" ]]; then
        echo "[ERROR] Cannot resume training from save-only-model checkpoints." >&2
        exit 1
    fi

    CHECKPOINT_RESUME_ARGS=()
    if [[ "${RESUME}" == "1" ]]; then
        if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
            checkpoint_resume_validate_target "${RESUME_FROM_CHECKPOINT}"
            RESUME_CHECKPOINT_DISPLAY="${RESUME_FROM_CHECKPOINT}"
            CHECKPOINT_RESUME_ARGS=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
        else
            latest_checkpoint="$(checkpoint_resume_find_latest "${output_dir}")"
            if [[ -z "${latest_checkpoint}" ]]; then
                echo "[ERROR] RESUME=1 but no resumable checkpoint-* found in OUTPUT_DIR=${output_dir}" >&2
                exit 1
            fi
            checkpoint_resume_validate_target "${latest_checkpoint}"
            RESUME_CHECKPOINT_DISPLAY="latest (${latest_checkpoint})"
            CHECKPOINT_RESUME_ARGS=(--resume-from-checkpoint "${latest_checkpoint}")
        fi
    else
        RESUME_CHECKPOINT_DISPLAY="-"
    fi

    CHECKPOINT_SAVE_ARGS=()
    SAVE_RESUMABLE_STATE=true
    if [[ "${SAVE_ONLY_MODEL}" == "1" ]]; then
        CHECKPOINT_SAVE_ARGS=(--save-only-model)
        SAVE_RESUMABLE_STATE=false
    fi

    export SAVE_STEPS SAVE_TOTAL_LIMIT RESUME RESUME_FROM_CHECKPOINT SAVE_ONLY_MODEL
}

checkpoint_resume_print_status() {
    local output_dir="$1"
    local resume_text=false
    if [[ "${RESUME}" == "1" ]]; then
        resume_text=true
    fi

    echo "Output directory      : ${output_dir}"
    echo "Save every            : ${SAVE_STEPS} steps"
    echo "Save total limit      : ${SAVE_TOTAL_LIMIT}"
    echo "Resume                 : ${resume_text}"
    echo "Resume checkpoint      : ${RESUME_CHECKPOINT_DISPLAY}"
    echo "Save resumable state   : ${SAVE_RESUMABLE_STATE}"
}

checkpoint_resume_require_fresh_output() {
    local output_dir="$1"
    if [[ "${RESUME}" == "0" && -d "${output_dir}" ]] && \
       [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
        echo "[ERROR] OUTPUT_DIR already exists and is not empty: ${output_dir}" >&2
        echo "        Set RESUME=1, choose a new OUTPUT_DIR, or use the timestamped default." >&2
        exit 1
    fi
}
