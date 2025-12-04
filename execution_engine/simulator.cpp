#include "orderbook.hpp"

#include <fstream>
#include <sstream>
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
// timestamp_ns,side,price,qty,order_type

SimResult run_simulation_csv(const std::string &path, const SimulationConfig &cfg) {
    OrderBook book;
    SimResult result;

    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string ts_s, side_s, price_s, qty_s, type_s, id_s;
        std::getline(ss, ts_s, ',');
        std::getline(ss, side_s, ',');
        std::getline(ss, price_s, ',');
        std::getline(ss, qty_s, ',');
        std::getline(ss, type_s, ',');
        std::getline(ss, id_s, ',');

        Order o;
        o.timestamp_ns = std::stoll(ts_s);
        o.side = (side_s == "BUY") ? Side::BUY : Side::SELL;
        o.price = std::stod(price_s);
        o.qty = std::stod(qty_s);
        o.id = id_s;

        std::vector<Trade> trades;
        if (type_s == "LIMIT") {
            book.add_limit(o, trades);
        } else {
            book.add_market(o, trades);
        }
        result.trades.insert(result.trades.end(), trades.begin(), trades.end());
    }
    return result;
}

} // namespace alpha


