function get_host_target() {
    local -r halide_root=$1
    local -n host_target_ref=$2

    echo "Calling get_host_target()..."
    host_target_ref=$(${AUTOSCHED_BIN}/get_host_target)
    echo "host_target = ${host_target_ref}"
    echo
}