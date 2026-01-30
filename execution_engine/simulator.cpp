#include "simulator.hpp"

#include <fstream>
#include <sstream>

namespace alpha {

SimResult run_simulation_csv(const std::string &path, const SimulationConfig &cfg) {
    MatchingEngine engine;
    SimResult result;

    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string ts_s, side_s, price_s, qty_s, type_s, id_s, symbol_s;
        std::getline(ss, ts_s, ',');
        std::getline(ss, side_s, ',');
        std::getline(ss, price_s, ',');
        std::getline(ss, qty_s, ',');
        std::getline(ss, type_s, ',');
        std::getline(ss, id_s, ',');
        std::getline(ss, symbol_s);

        if (!symbol_s.empty() && symbol_s.back() == '\r') symbol_s.pop_back();

        Order o;
        o.timestamp_ns = std::stoll(ts_s);
        o.side = (side_s == "BUY") ? Side::BUY : Side::SELL;
        o.price = std::stod(price_s);
        o.qty = std::stod(qty_s);
        o.id = id_s;
        o.symbol = symbol_s;

        std::vector<Trade> trades;
        if (type_s == "LIMIT") {
            engine.add_limit(o, trades);
        } else {
            engine.add_market(o, trades);
        }
        result.trades.insert(result.trades.end(), trades.begin(), trades.end());
    }
    return result;
}

} // namespace alpha


