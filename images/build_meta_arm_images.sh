#!/usr/bin/env bash
# Build the episode images the kept forms name, and print each image's digest.
#
#   python    `rqgm-python-family:t1`: python 3.14 plus the pytest closure. The image of
#             toy's generation episodes (`image_recipe: meta-arm/python`).
#   mini-swe  the python family plus the mini-swe closure and ripgrep. The image of every
#             form's expansion episode and of the imo/paper generation episodes
#             (`image_recipe: meta-arm/mini-swe`).
#   runtime   python 3.14 and the mini-swe closure under `/opt/agent-runtime`, relocatable, for
#             linux/amd64 always. Episodes that work inside a sample's own image (deepswe's
#             solver) get this directory copied into that image (`image_recipe: meta-arm/runtime`).
#
# A run resolves a recipe by asking this script for its tag (`--reference`) and reading
# the digest the local daemon holds under that tag.
#
#   images/build_meta_arm_images.sh                                 # both stanzas
#   images/build_meta_arm_images.sh --platform linux/amd64 mini-swe
#   images/build_meta_arm_images.sh python --reference              # print the tag only
#
# `--platform` defaults to the daemon's native platform. Environment:
#   RQGM_PIP_INDEX_URL  the package index wheels are downloaded from (default public PyPI)
#   RQGM_WHEELHOUSE     a directory of pre-downloaded wheels, for a host with no egress
#   RQGM_BUILD_COLD=1   build with `--no-cache`
#
# Every package is installed `--no-deps` at a spelled-out version, and every timestamp
# that would otherwise reach a layer or the image config is fixed, so two cold builds
# of the same recipe produce the same digest.
set -euo pipefail

platform=""
reference_only=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --platform)
      test "$#" -ge 2 || { printf -- '--platform needs a value\n' >&2; exit 1; }
      platform="$2"; shift 2 ;;
    --platform=*) platform="${1#--platform=}"; shift ;;
    --reference) reference_only="yes"; shift ;;
    *) break ;;
  esac
done
# `--reference` may also follow the stanza names.
for _arg in "$@"; do
  [ "$_arg" = "--reference" ] && reference_only="yes"
done
# `--reference` needs no daemon.
if [ -z "$platform" ] && [ -z "$reference_only" ]; then
  platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
fi

pip_index="${RQGM_PIP_INDEX_URL:-https://pypi.org/simple}"

cold_args=()
if [ -n "${RQGM_BUILD_COLD:-}" ]; then
  cold_args=(--no-cache)
fi

# Download the closure on the host into $wheelhouse, as binary wheels for the target
# platform, so the build itself needs no network.
wheelhouse=""
stage() {
  # Args: the closure to fetch. Fills $wheelhouse.
  local arch="${platform#*/}" tag
  case "$arch" in
    arm64 | aarch64) tag="manylinux2014_aarch64" ;;
    amd64 | x86_64) tag="manylinux2014_x86_64" ;;
    *) printf 'no wheel platform tag known for %s\n' "$arch" >&2; exit 1 ;;
  esac
  wheelhouse="$ctx/wheels-$arch"
  mkdir -p "$wheelhouse"
  if [ -n "${RQGM_WHEELHOUSE:-}" ]; then
    cp -R "${RQGM_WHEELHOUSE}/." "$wheelhouse/"
  else
    python3 -m pip download --no-deps --only-binary=:all: \
      --python-version 3.14 --platform "$tag" --index-url "$pip_index" \
      --retries 10 --timeout 120 \
      -d "$wheelhouse" "$@" >&2 || {
        printf 'the host could not fetch the closure for %s; fill RQGM_WHEELHOUSE\n' "$tag" >&2
        exit 1
      }
  fi
  # BuildKit hashes file mtimes into a COPY layer, so the wheels get a fixed one.
  find "$wheelhouse" -exec touch -d @0 {} +
}

# The install step both stanzas run. pip stamps unpacked files with the wall clock and
# timestamp-invalidated `.pyc` headers embed those mtimes (PEP 552), so everything the
# step created is set to epoch 0 and the bytecode is rebuilt hash-invalidated. The
# marker file limits the restamp to paths this step touched. It must stay one `RUN`:
# a layer is the unit a digest is taken over.
install_layer() {
  # Args: the closure to install.
  cat <<INSTALL_EOF
RUN set -eu; \\
    touch /tmp/.rqgm-epoch; \\
    pip install --no-cache-dir --no-deps --no-index --find-links=/wheels $*; \\
    rm -rf /wheels; \\
    site="\$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \\
    find "\$site" -name __pycache__ -prune -exec rm -rf {} +; \\
    find /usr/local -newer /tmp/.rqgm-epoch -exec touch -hd @0 {} +; \\
    python3 -m compileall -q -f --invalidation-mode unchecked-hash "\$site"; \\
    find /usr/local -newer /tmp/.rqgm-epoch -exec touch -hd @0 {} +; \\
    rm -f /tmp/.rqgm-epoch
INSTALL_EOF
}

# Build one image with buildx and print its ID. `docker build` without the buildx plugin
# accepts `--platform` and ignores it, so the architecture of the result is checked.
#
# `SOURCE_DATE_EPOCH` (read by buildx from the environment) fixes the config's `created`
# field and the history timestamps; `rewrite-timestamp=true` fixes the mtimes inside the
# layer tars and needs `unpack=false`; `--provenance=false` drops the attestation
# manifest, which records the build's start time. The status is captured rather than
# left to `set -e` so the checks below also run after a failed build, which can still
# leave a tagged image behind.
build() {
  local tag="$1" ctx_dir="$2" got want="${platform#*/}" before now status=0
  shift 2
  before="$(docker image inspect --format '{{.Id}}' "$tag" 2>/dev/null || true)"
  SOURCE_DATE_EPOCH=1735689600 docker buildx build \
    --output "type=docker,unpack=false,rewrite-timestamp=true" \
    --provenance=false \
    "${cold_args[@]+"${cold_args[@]}"}" \
    --platform "$platform" -t "$tag" "$@" "$ctx_dir" >&2 || status=$?

  got="$(docker image inspect --format '{{.Architecture}}' "$tag" 2>/dev/null || true)"
  if [ -z "$got" ]; then
    printf 'nothing to check: %s does not resolve to an image\n' "$tag" >&2
    exit 1
  fi
  printf 'checked %s: architecture %s (want %s)\n' "$tag" "$got" "$want" >&2
  if [ "$got" != "$want" ]; then
    printf 'asked for %s and built %s; install the buildx plugin\n' \
      "$platform" "$got" >&2
    exit 1
  fi
  if [ "$status" -ne 0 ]; then
    printf 'the build command exited %d\n' "$status" >&2
    now="$(docker image inspect --format '{{.Id}}' "$tag" 2>/dev/null || true)"
    if [ -n "$now" ] && [ "$now" != "$before" ]; then
      printf 'and yet %s now names %s, which this run produced: the check above\n' \
        "$tag" "$now" >&2
      printf 'describes that image. Do not pin it -- a build that reported\n' >&2
      printf 'failure has not earned a pin.\n' >&2
    fi
    exit 1
  fi
  docker image inspect --format '{{.Id}}' "$tag"
}

STANZAS="python mini-swe runtime"
selection=""
for name in "$@"; do
  case "$name" in
    --reference) reference_only="yes" ;;
    *) selection="$selection $name" ;;
  esac
done
requested="${selection:-$STANZAS}"
for name in $requested; do
  case " $STANZAS " in
    *" $name "*) ;;
    *) printf 'no such stanza: %s (have: %s)\n' "$name" "$STANZAS" >&2; exit 1 ;;
  esac
done

selected() {
  case " $requested " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

# The tag each stanza applies, read by both `--reference` and the build.
TAG_python="rqgm-python-family:t1"
TAG_mini_swe="rqgm-mini-swe:t1"
TAG_runtime="rqgm-runtime:t1"

if [ -n "$reference_only" ]; then
  for name in $requested; do
    case "$name" in
      python) printf '%s\n' "$TAG_python" ;;
      mini-swe) printf '%s\n' "$TAG_mini_swe" ;;
      runtime) printf '%s\n' "$TAG_runtime" ;;
    esac
  done
  exit 0
fi

PY_BASE="python:3.14-bookworm@sha256:5dcba30b5f8fbd97e2f35dd1b140b3c94db70bd01b39ed88365732f8db8f68b5"
# The pytest closure every python-family episode image carries.
PYTEST_CLOSURE="pytest==9.1.1 iniconfig==2.3.0 packaging==26.2 pluggy==1.6.0 pygments==2.20.0"
# The vendored mini-swe agent's closure plus pytest, in the mini-swe image and the agent runtime.
MINI_SWE_CLOSURE="jinja2==3.1.6 markupsafe==3.0.3 pydantic==2.13.4 pydantic-core==2.46.4 annotated-types==0.7.0 typing-extensions==4.15.0 typing-inspection==0.4.2 ${PYTEST_CLOSURE}"

ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT

# -- python: the python base plus the pytest closure ---------------------------
if selected python; then
mkdir -p "$ctx/family-ctx"
stage ${PYTEST_CLOSURE}
cp -Rp "$wheelhouse" "$ctx/family-ctx/wheels"
cat > "$ctx/family-ctx/Dockerfile" <<EOF
FROM ${PY_BASE}
COPY wheels /wheels
$(install_layer ${PYTEST_CLOSURE})
EOF
family_digest="$(build "$TAG_python" "$ctx/family-ctx")"
echo "PINNED_PYTHON = $family_digest"
fi

# -- mini-swe: the python base, the mini-swe closure, the pytest closure, ripgrep --
# The closure is each form's `genesis/requirements.lock` plus the pytest closure; if
# they disagree, the lock file is right. ripgrep comes from Debian because `rg` is the
# search tool most coding agents reach for first.
if selected mini-swe; then
mkdir -p "$ctx/ms-ctx"
stage ${MINI_SWE_CLOSURE}
cp -Rp "$wheelhouse" "$ctx/ms-ctx/wheels"
cat > "$ctx/ms-ctx/Dockerfile" <<EOF
FROM ${PY_BASE}
COPY wheels /wheels
$(install_layer ${MINI_SWE_CLOSURE})
# The same epoch treatment the install layer gets, and for the same reason. With
# the wheelhouse, the bytecode and the image config all normalised, two COLD
# builds of this stanza differed in EXACTLY ONE of eleven layers -- `34c8e691`
# against `6b4f28e6` -- and it was this one. `dpkg` stamps what it unpacks with
# the wall clock and writes its own status and log files as it goes, so an
# `apt-get install` layer carries the second it ran no matter how fixed the
# package version is. The python stanza has no such layer, which is why it
# reproduced (`052c0981` twice) while this one did not.
#
# The marker keeps it surgical, as above: only paths this step created or touched
# are stamped, so nothing from the base image is dragged up into this layer. The
# apt lists are already deleted here; the dpkg log is truncated rather than
# removed because dpkg owns the path and expects it to exist.
RUN set -eu; \\
    touch /tmp/.rqgm-epoch; \\
    apt-get update; \\
    apt-get install -y --no-install-recommends ripgrep; \\
    rm -rf /var/lib/apt/lists/*; \\
    : > /var/log/dpkg.log; \\
    : > /var/log/alternatives.log; \\
    : > /var/log/apt/history.log; \\
    : > /var/log/apt/term.log; \\
    : > /var/log/apt/eipp.log.xz; \\
    find / -xdev -newer /tmp/.rqgm-epoch -not -path '/proc/*' -not -path '/sys/*' \\
      -not -path '/tmp/.rqgm-epoch' -exec touch -hd @0 {} +; \\
    rm -f /tmp/.rqgm-epoch
RUN rg --version
EOF
ms_digest="$(build "$TAG_mini_swe" "$ctx/ms-ctx")"
echo "PINNED_MINI_SWE_RUNTIME = $ms_digest"
fi

# -- runtime: a relocatable interpreter under /opt/agent-runtime, for amd64 sample images --
# The interpreter finds its standard library beside it, so it runs in an image that has never
# heard of it. Sample images are amd64, so this stanza is always built for amd64.
if selected runtime; then
  platform="linux/amd64"
  mkdir -p "$ctx/rt-ctx"
  stage ${MINI_SWE_CLOSURE}
  cp -Rp "$wheelhouse" "$ctx/rt-ctx/wheels"
  cat > "$ctx/rt-ctx/Dockerfile" <<EOF
FROM ${PY_BASE} AS build
COPY wheels /wheels
RUN set -eu; \\
    install -d /opt/agent-runtime/bin /opt/agent-runtime/lib; \\
    cp -a /usr/local/bin/python3.14 /opt/agent-runtime/bin/; \\
    ln -s python3.14 /opt/agent-runtime/bin/python3; ln -s python3 /opt/agent-runtime/bin/python; \\
    cp -a /usr/local/lib/libpython3.14.so* /usr/local/lib/python3.14 /opt/agent-runtime/lib/; \\
    PYTHONHOME=/opt/agent-runtime /opt/agent-runtime/bin/python -m pip install --no-cache-dir --no-deps \\
      --no-index --find-links=/wheels --no-compile ${MINI_SWE_CLOSURE}; \\
    find /opt/agent-runtime -name __pycache__ -prune -exec rm -rf {} +; \\
    PYTHONHOME=/opt/agent-runtime /opt/agent-runtime/bin/python -m compileall -q -f \\
      --invalidation-mode unchecked-hash /opt/agent-runtime/lib/python3.14; \\
    find /opt/agent-runtime -exec touch -hd @0 {} +
FROM scratch
COPY --from=build /opt/agent-runtime /opt/agent-runtime
EOF
  rt_digest="$(build "$TAG_runtime" "$ctx/rt-ctx")"
  echo "PINNED_AGENT_RUNTIME = $rt_digest"
fi
