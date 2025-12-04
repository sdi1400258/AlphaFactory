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
    SimulationConfig cfg; // use defaults

    std::cout << "Running simulation from CSV: " << path << "\n";
    SimResult res = run_simulation_csv(path, cfg);

    double notional = 0.0;
    for (const auto &t : res.trades) {
        notional += t.price * t.qty;
    }

    std::cout << "Trades executed: " << res.trades.size() << "\n";
    std::cout << "Total traded notional: " << notional << "\n";
    return 0;
}


