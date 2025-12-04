#pragma once

#include "orderbook.hpp"

#include <string>
#include <vector>

namespace alpha {

struct SimulationConfig {
    double max_notional = 1e6;
    double max_abs_position = 1000.0;
    double latency_ms = 5.0;
};

struct SimResult {
    std::vector<Trade> trades;
};

// Very lightweight CSV-driven simulator:
// timestamp_ns,side,price,qty,order_type,id
SimResult run_simulation_csv(const std::string &path, const SimulationConfig &cfg);

} // namespace alpha


