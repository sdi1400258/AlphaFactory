#include "simulator.hpp"

#include <iostream>
#include <string>

int main(int argc, char **argv) {
    using namespace alpha;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " ORDERS_CSV_PATH\n";
        return 1;
    }

    std::string path = argv[1];
    SimulationConfig cfg; 

    SimResult res = run_simulation_csv(path, cfg);

    // Output results as a simple JSON list for Python
    std::cout << "[\n";
    for (size_t i = 0; i < res.trades.size(); ++i) {
        const auto &t = res.trades[i];
        std::cout << "  {\"symbol\": \"" << t.symbol
                  << "\", \"taker_id\": \"" << t.taker_id 
                  << "\", \"maker_id\": \"" << t.maker_id
                  << "\", \"side\": \"" << ((t.side == Side::BUY) ? "BUY" : "SELL")
                  << "\", \"price\": " << t.price
                  << ", \"qty\": " << t.qty
                  << ", \"timestamp\": " << t.timestamp_ns << "}";
        if (i < res.trades.size() - 1) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "]\n";

    return 0;
}


