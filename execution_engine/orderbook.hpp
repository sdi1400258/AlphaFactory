#pragma once

#include <deque>
#include <string>
#include <vector>

namespace alpha {

enum class Side { BUY, SELL };

struct Order {
    std::string id;
    Side side;
    double price;
    double qty;
    long long timestamp_ns;
};

struct Trade {
    std::string taker_id;
    std::string maker_id;
    Side side; // side of taker
    double price;
    double qty;
    long long timestamp_ns;
};

class PriceLevel {
  public:
    double price;
    std::deque<Order> queue;

    explicit PriceLevel(double p) : price(p) {}
};

class OrderBook {
  public:
    std::vector<PriceLevel> bids; // sorted descending
    std::vector<PriceLevel> asks; // sorted ascending

    void add_limit(const Order &o, std::vector<Trade> &trades);
    void add_market(const Order &o, std::vector<Trade> &trades);
    void cancel(const std::string &order_id);

    double best_bid() const;
    double best_ask() const;
    double mid() const;
};

} // namespace alpha


