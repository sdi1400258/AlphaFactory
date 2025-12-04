#include "orderbook.hpp"

#include <algorithm>

namespace alpha {

namespace {

PriceLevel *find_level(std::vector<PriceLevel> &levels, double price, bool is_bid) {
    auto it = std::find_if(levels.begin(), levels.end(),
                           [price](const PriceLevel &lvl) { return lvl.price == price; });
    if (it != levels.end())
        return &(*it);

    // insert maintaining sort
    PriceLevel lvl(price);
    levels.push_back(lvl);
    std::sort(levels.begin(), levels.end(), [is_bid](const PriceLevel &a, const PriceLevel &b) {
        return is_bid ? a.price > b.price : a.price < b.price;
    });
    return find_level(levels, price, is_bid);
}

} // namespace

void OrderBook::add_limit(const Order &o, std::vector<Trade> &trades) {
    if (o.side == Side::BUY) {
        // match against best asks
        for (auto it = asks.begin(); it != asks.end() && o.price >= it->price;) {
            auto &lvl = *it;
            for (auto qit = lvl.queue.begin(); qit != lvl.queue.end() && o.qty > 0;) {
                double traded = std::min(o.qty, qit->qty);
                trades.push_back(
                    {o.id, qit->id, Side::BUY, lvl.price, traded, o.timestamp_ns});
                o.qty -= traded;
                qit->qty -= traded;
                if (qit->qty <= 0) {
                    qit = lvl.queue.erase(qit);
                } else {
                    ++qit;
                }
            }
            if (lvl.queue.empty()) {
                it = asks.erase(it);
            } else {
                ++it;
            }
            if (o.qty <= 0)
                return;
        }
        if (o.qty > 0) {
            Order resting = o;
            auto *lvl = find_level(bids, resting.price, true);
            lvl->queue.push_back(resting);
        }
    } else {
        // SELL: match against best bids
        for (auto it = bids.begin(); it != bids.end() && o.price <= it->price;) {
            auto &lvl = *it;
            for (auto qit = lvl.queue.begin(); qit != lvl.queue.end() && o.qty > 0;) {
                double traded = std::min(o.qty, qit->qty);
                trades.push_back(
                    {o.id, qit->id, Side::SELL, lvl.price, traded, o.timestamp_ns});
                o.qty -= traded;
                qit->qty -= traded;
                if (qit->qty <= 0) {
                    qit = lvl.queue.erase(qit);
                } else {
                    ++qit;
                }
            }
            if (lvl.queue.empty()) {
                it = bids.erase(it);
            } else {
                ++it;
            }
            if (o.qty <= 0)
                return;
        }
        if (o.qty > 0) {
            Order resting = o;
            auto *lvl = find_level(asks, resting.price, false);
            lvl->queue.push_back(resting);
        }
    }
}

void OrderBook::add_market(const Order &o_in, std::vector<Trade> &trades) {
    Order o = o_in;
    if (o.side == Side::BUY) {
        for (auto it = asks.begin(); it != asks.end() && o.qty > 0;) {
            auto &lvl = *it;
            for (auto qit = lvl.queue.begin(); qit != lvl.queue.end() && o.qty > 0;) {
                double traded = std::min(o.qty, qit->qty);
                trades.push_back(
                    {o.id, qit->id, Side::BUY, lvl.price, traded, o.timestamp_ns});
                o.qty -= traded;
            qit->qty -= traded;
                if (qit->qty <= 0) {
                    qit = lvl.queue.erase(qit);
                } else {
                    ++qit;
                }
            }
            if (lvl.queue.empty()) {
                it = asks.erase(it);
            } else {
                ++it;
            }
        }
    } else {
        for (auto it = bids.begin(); it != bids.end() && o.qty > 0;) {
            auto &lvl = *it;
            for (auto qit = lvl.queue.begin(); qit != lvl.queue.end() && o.qty > 0;) {
                double traded = std::min(o.qty, qit->qty);
                trades.push_back(
                    {o.id, qit->id, Side::SELL, lvl.price, traded, o.timestamp_ns});
                o.qty -= traded;
                qit->qty -= traded;
                if (qit->qty <= 0) {
                    qit = lvl.queue.erase(qit);
                } else {
                    ++qit;
                }
            }
            if (lvl.queue.empty()) {
                it = bids.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void OrderBook::cancel(const std::string &order_id) {
    auto cancel_from = [&order_id](std::vector<PriceLevel> &levels) {
        for (auto it = levels.begin(); it != levels.end();) {
            auto &lvl = *it;
            for (auto qit = lvl.queue.begin(); qit != lvl.queue.end();) {
                if (qit->id == order_id) {
                    qit = lvl.queue.erase(qit);
                } else {
                    ++qit;
                }
            }
            if (lvl.queue.empty()) {
                it = levels.erase(it);
            } else {
                ++it;
            }
        }
    };

    cancel_from(bids);
    cancel_from(asks);
}

double OrderBook::best_bid() const { return bids.empty() ? 0.0 : bids.front().price; }

double OrderBook::best_ask() const { return asks.empty() ? 0.0 : asks.front().price; }

double OrderBook::mid() const {
    if (bids.empty() || asks.empty())
        return 0.0;
    return 0.5 * (best_bid() + best_ask());
}

} // namespace alpha


