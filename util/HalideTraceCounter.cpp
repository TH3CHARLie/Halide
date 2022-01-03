#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <fstream>

#include <unistd.h>

#include "HalideRuntime.h"

#include "halide_trace_config.h"

using namespace Halide;
using namespace Halide::Trace;
bool verbose = false;

std::map<std::string, int> vector_load_counters;
std::map<std::string, int> scalar_load_counters;
std::map<std::string, int> store_counters;

// Log informational output to stderr, but only in verbose mode
struct info {
    std::ostringstream msg;

    template<typename T>
    info &operator<<(const T &x) {
        if (verbose) {
            msg << x;
        }
        return *this;
    }

    ~info() {
        if (verbose) {
            if (msg.str().back() != '\n') {
                msg << "\n";
            }
            std::cerr << msg.str();
        }
    }
};

// Log warnings to stderr
struct warn {
    std::ostringstream msg;

    template<typename T>
    warn &operator<<(const T &x) {
        msg << x;
        return *this;
    }

    ~warn() {
        if (msg.str().back() != '\n') {
            msg << "\n";
        }
        std::cerr << "Warning: " << msg.str();
    }
};

// Log unrecoverable errors to stderr, then exit
struct fail {
    std::ostringstream msg;

    template<typename T>
    fail &operator<<(const T &x) {
        msg << x;
        return *this;
    }

    ~fail() {
        if (msg.str().back() != '\n') {
            msg << "\n";
        }
        std::cerr << msg.str();
        exit(1);
    }
};

struct PacketAndPayload : public halide_trace_packet_t {
    uint8_t payload[4096];

    static bool read_or_die(void *buf, size_t count) {
        char *p = (char *)buf;
        char *p_end = p + count;
        while (p < p_end) {
            int64_t bytes_read = ::read(STDIN_FILENO, p, p_end - p);
            if (bytes_read == 0) {
                return false;  // EOF
            } else if (bytes_read < 0) {
                fail() << "Unable to read packet";
            }
            p += bytes_read;
        }
        assert(p == p_end);
        return true;
    }

    bool read() {
        constexpr size_t header_size = sizeof(halide_trace_packet_t);
        if (!read_or_die(this, header_size)) {
            return false;  // EOF
        }

        const size_t payload_size = this->size - header_size;
        if (payload_size > sizeof(this->payload) || !read_or_die(this->payload, payload_size)) {
            // Shouldn't ever get EOF here
            fail() << "Unable to read packet payload of size " << payload_size;
            return false;
        }
        return true;
    }
};

int run() {
    struct PipelineInfo {
        std::string name;
        int32_t id;
    };
    std::map<uint32_t, PipelineInfo> pipeline_info;
    bool flag = false;
    while (true) {
        PacketAndPayload p;
        if (!p.read()) {
          break;
        }

        if (p.event == halide_trace_begin_pipeline) {
            if (flag) {
                break;
            }
            pipeline_info[p.id] = {p.func(), p.id};
            flag = true;
            continue;
        } else if (p.event == halide_trace_end_pipeline) {
            assert(pipeline_info.count(p.parent_id));
            pipeline_info.erase(p.parent_id);
            continue;
        }

        const PipelineInfo pipeline = pipeline_info[p.parent_id];

        switch (p.event) {
            case halide_trace_load: {
                std::string func_name = p.func();
                if (p.type.lanes > 1) {
                    if (vector_load_counters.count(func_name) == 0) {
                        vector_load_counters[func_name] = p.type.lanes;
                    } else {
                        vector_load_counters[func_name] += p.type.lanes;
                    }
                } else {
                    if (scalar_load_counters.count(func_name) == 0) {
                        scalar_load_counters[func_name] = 1;
                    } else {
                        scalar_load_counters[func_name]++;
                    }
                }
                break;
            }
            case halide_trace_store: {
                std::string func_name = p.func();
                if (p.type.lanes > 1) {
                    if (store_counters.count(func_name) == 0) {
                        store_counters[func_name] = p.type.lanes;
                    } else {
                        store_counters[func_name] += p.type.lanes;
                    }
                } else {
                    if (store_counters.count(func_name) == 0) {
                        store_counters[func_name] = 1;
                    } else {
                        store_counters[func_name]++;
                    }
                }
                break;
            }
        }
    }
    return 0;
}


void dump_to_file(char *filename) {
    std::ofstream out(filename);
    out << "Vector Load Counters:\n";
    for (const auto& x : vector_load_counters) {
        out << x.first << ": " << x.second << '\n';
    }
    out << "Scalar Load Counters:\n";
    for (const auto& x : scalar_load_counters) {
        out << x.first << ": " << x.second << '\n';
    }
    out << "Store Counters:\n";
    for (const auto& x : store_counters) {
        out << x.first << ": " << x.second << '\n';
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        return 1;
    }
    run();
    dump_to_file(argv[1]);
    return 0;
}
