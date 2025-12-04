#include "orderbook.hpp"

namespace alpha {

bool basic_notional_check(double notional, double max_notional) {
    return notional <= max_notional;
}

bool max_position_check(double current_pos, double delta, double max_abs_pos) {
    double new_pos = current_pos + delta;
    return std::abs(new_pos) <= max_abs_pos;
}

} // namespace alpha


