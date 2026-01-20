//! Market Module
//!
//! This module provides the order book and matching engine for the BTUT simulator.
//!
//! ## Components
//!
//! - `OrderBook`: Central limit order book with price-time priority matching
//! - `Order`: Individual order with price, quantity, and metadata
//! - `Trade`: Executed trade result
//!
//! ## Features
//!
//! - Efficient price level management using sorted maps
//! - FIFO matching within price levels
//! - Support for limit and market orders
//! - Order cancellation and modification
//! - Snapshot and depth queries

mod order_book;

pub use order_book::{
    BookSnapshot, Execution, Order, OrderBook, OrderId, OrderSide, OrderType, Price, PriceLevel,
    Quantity, Trade,
};
