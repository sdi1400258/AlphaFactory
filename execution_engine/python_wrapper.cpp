#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "orderbook.hpp"

namespace py = pybind11;

PYBIND11_MODULE(alpha_engine, m) {
    m.doc() = "AlphaFactory C++ Execution Engine Wrapper";

    py::enum_<alpha::Side>(m, "Side")
        .value("BUY", alpha::Side::BUY)
        .value("SELL", alpha::Side::SELL)
        .export_values();

    py::class_<alpha::Order>(m, "Order")
        .def(py::init<std::string, alpha::Side, double, double, long long>(),
             py::arg("id"), py::arg("side"), py::arg("price"), py::arg("qty"), py::arg("timestamp_ns"))
        .def_readwrite("id", &alpha::Order::id)
        .def_readwrite("side", &alpha::Order::side)
        .def_readwrite("price", &alpha::Order::price)
        .def_readwrite("qty", &alpha::Order::qty)
        .def_readwrite("timestamp_ns", &alpha::Order::timestamp_ns);

    py::class_<alpha::Trade>(m, "Trade")
        .def_readwrite("taker_id", &alpha::Trade::taker_id)
        .def_readwrite("maker_id", &alpha::Trade::maker_id)
        .def_readwrite("side", &alpha::Trade::side)
        .def_readwrite("price", &alpha::Trade::price)
        .def_readwrite("qty", &alpha::Trade::qty)
        .def_readwrite("timestamp_ns", &alpha::Trade::timestamp_ns);

    py::class_<alpha::OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("add_limit", &alpha::OrderBook::add_limit)
        .def("add_market", &alpha::OrderBook::add_market)
        .def("cancel", &alpha::OrderBook::cancel)
        .def("best_bid", &alpha::OrderBook::best_bid)
        .def("best_ask", &alpha::OrderBook::best_ask)
        .def("mid", &alpha::OrderBook::mid);
}
