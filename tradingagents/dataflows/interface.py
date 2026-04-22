from typing import Annotated, Optional

# Import from vendor-specific modules
from .y_finance import (
    get_YFin_data_online, 
    get_indicators as get_yfinance_indicators, 
    get_fundamentals as get_yfinance_fundamentals, 
    get_balance_sheet as get_yfinance_balance_sheet, 
    get_cashflow as get_yfinance_cashflow, 
    get_income_statement as get_yfinance_income_statement, 
    get_insider_transactions as get_yfinance_insider_transactions, 
    get_market_context as get_yfinance_market_context
)
from .vnstock_finance import (
    get_stock_data as get_vnstock_stock_data, 
    get_indicators as get_vnstock_indicators, 
    get_fundamentals as get_vnstock_fundamentals, 
    get_balance_sheet as get_vnstock_balance_sheet, 
    get_cashflow as get_vnstock_cashflow, 
    get_income_statement as get_vnstock_income_statement, 
    get_news as get_vnstock_news, 
    get_market_context as get_vnstock_market_context
)
from .cafef_news import get_cafef_news
from .vietstock_news import get_vietstock_global_news
from .openai import get_stock_news_openai, get_global_news_openai, get_fundamentals_openai
from .social_media import get_f247_forum_posts, get_ticker_news

# Configuration and routing logic
from .config import get_config

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data",
            "get_market_context",
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement",
        ]
    },
    "news_data": {
        "description": "News (public/insiders, original/processed)",
        "tools": [
            "get_news",
        ]
    },
    "social_data": {
        "description": "Social/forum discussion and ticker-specific social news",
        "tools": [
            "get_f247_forum_posts",
            "get_ticker_news",
        ]
    },
    "global_data": {
        "description": "Global macroeconomic data and news",
        "tools": [
            "get_global_news",
        ]
    },
    "insider_transaction_data": {
        "description": "Insider transaction data",
        "tools": [
            "get_insider_transactions"
        ]
    }
}

VENDOR_LIST = [
    "vnstock",
    "yfinance",
    "vietstock",
    "openai",
    "cafef",
    "f247",
]


def _is_error_result(result) -> bool:
    """Heuristic to detect stringified vendor failures that should trigger fallback."""
    if not isinstance(result, str):
        return False

    text = result.strip()
    if not text:
        return True

    lowered = text.lower()

    # Common explicit failure prefixes returned by vendor wrappers.
    if lowered.startswith("lỗi") or lowered.startswith("error") or lowered.startswith("failed"):
        return True

    # Network/retry/runtime error signatures often surfaced as plain strings.
    error_signatures = [
        "retryerror[",
        "connectionerror",
        "traceback (most recent call last)",
        "exception:",
        "runtimeerror",
        "timed out",
    ]
    return any(sig in lowered for sig in error_signatures)

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "vnstock": get_vnstock_stock_data,
        "yfinance": get_YFin_data_online,
    },
    "get_market_context": {
        "vnstock": get_vnstock_market_context,
        "yfinance": get_yfinance_market_context,
    },
    # technical_indicators
    "get_indicators": {
        "vnstock": get_vnstock_indicators,
        "yfinance": get_yfinance_indicators,
    },
    # fundamental_data
    "get_fundamentals": {
        "vnstock": get_vnstock_fundamentals,
        "yfinance": get_yfinance_fundamentals,
        "openai": get_fundamentals_openai,
    },
    "get_balance_sheet": {
        "vnstock": get_vnstock_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
    },
    "get_cashflow": {
        "vnstock": get_vnstock_cashflow,
        "yfinance": get_yfinance_cashflow,
    },
    "get_income_statement": {
        "vnstock": get_vnstock_income_statement,
        "yfinance": get_yfinance_income_statement,
    },
    # news_data
    "get_news": {
        "vnstock": get_vnstock_news,
        "cafef": get_cafef_news,
        "openai": get_stock_news_openai,
    },
    "get_global_news": {
        "vietstock": get_vietstock_global_news,
        "openai": get_global_news_openai,
    },
    # social_data
    "get_f247_forum_posts": {
        "f247": get_f247_forum_posts,
    },
    "get_ticker_news": {
        "f247": get_ticker_news,
    },
    # insider_transaction_data
    "get_insider_transactions": {
        "yfinance": get_yfinance_insider_transactions,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: Optional[str] = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    vendor = config.get("data_vendors", {}).get(category, "default")
    return vendor if isinstance(vendor, str) else "default"

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)

    # Handle comma-separated vendors
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Get all available vendors for this method for fallback
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    
    # Create fallback vendor list: primary vendors first, then remaining vendors as fallbacks
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    # Debug: Print fallback ordering
    primary_str = " -> ".join(primary_vendors)
    fallback_str = " -> ".join(fallback_vendors)
    print(f"DEBUG: {method} - Primary: [{primary_str}] | Full fallback order: [{fallback_str}]")

    # Track results and execution state
    results = []
    vendor_attempt_count = 0
    any_primary_vendor_attempted = False
    successful_vendor = None

    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            if vendor in primary_vendors:
                print(f"INFO: Vendor '{vendor}' not supported for method '{method}', falling back to next vendor")
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        is_primary_vendor = vendor in primary_vendors
        vendor_attempt_count += 1

        # Track if we attempted any primary vendor
        if is_primary_vendor:
            any_primary_vendor_attempted = True

        # Debug: Print current attempt
        vendor_type = "PRIMARY" if is_primary_vendor else "FALLBACK"
        print(f"DEBUG: Attempting {vendor_type} vendor '{vendor}' for {method} (attempt #{vendor_attempt_count})")

        # Handle list of methods for a vendor
        if isinstance(vendor_impl, list):
            vendor_methods = [(impl, vendor) for impl in vendor_impl]
            print(f"DEBUG: Vendor '{vendor}' has multiple implementations: {len(vendor_methods)} functions")
        else:
            vendor_methods = [(vendor_impl, vendor)]

        # Run methods for this vendor
        vendor_results = []
        for impl_func, vendor_name in vendor_methods:
            try:
                print(f"DEBUG: Calling {impl_func.__name__} from vendor '{vendor_name}'...")
                result = impl_func(*args, **kwargs)
                if result is None:
                    print(f"FAILED: {impl_func.__name__} from vendor '{vendor_name}' returned None")
                    continue
                if isinstance(result, str) and not result.strip():
                    print(f"FAILED: {impl_func.__name__} from vendor '{vendor_name}' returned empty string")
                    continue
                if _is_error_result(result):
                    print(
                        f"FAILED: {impl_func.__name__} from vendor '{vendor_name}' returned error-like result: {str(result)[:220]}"
                    )
                    continue
                vendor_results.append(result)
                print(f"SUCCESS: {impl_func.__name__} from vendor '{vendor_name}' completed successfully")
                    
            except Exception as e:
                # Log error but continue with other implementations
                print(f"FAILED: {impl_func.__name__} from vendor '{vendor_name}' failed: {e}")
                continue

        # Add this vendor's results
        if vendor_results:
            results.extend(vendor_results)
            successful_vendor = vendor
            result_summary = f"Got {len(vendor_results)} result(s)"
            print(f"SUCCESS: Vendor '{vendor}' succeeded - {result_summary}")

            # For news, we want strict fallback behavior:
            # try the next vendor only when current one fails/empty, and stop at first success.
            if method == "get_news":
                print(f"DEBUG: Stopping after successful vendor '{vendor}' for get_news")
                break

            # Stopping logic: Stop after first successful vendor for single-vendor configs.
            # Multiple vendor configs (comma-separated) may want to collect from multiple sources.
            if len(primary_vendors) == 1:
                print(f"DEBUG: Stopping after successful vendor '{vendor}' (single-vendor config)")
                break
        else:
            print(f"FAILED: Vendor '{vendor}' produced no results")

    # Final result summary
    if not results:
        print(f"FAILURE: All {vendor_attempt_count} vendor attempts failed for method '{method}'")
        if primary_vendors and not any_primary_vendor_attempted:
            print("INFO: None of the configured primary vendors were supported; only fallbacks were attempted.")
        raise RuntimeError(f"All vendor implementations failed for method '{method}'")
    else:
        print(
            f"FINAL: Method '{method}' completed with {len(results)} result(s) "
            f"from {vendor_attempt_count} vendor attempt(s); successful vendor: {successful_vendor}"
        )

    # Return single result if only one, otherwise concatenate as string
    if len(results) == 1:
        return results[0]
    else:
        # Convert all results to strings and concatenate
        return '\n'.join(str(result) for result in results)

if __name__ == "__main__":
    # Example usage
    try:
        stock_data = route_to_vendor("get_cashflow", symbol="HPG", period="quater")
        print(stock_data)
    except Exception as e:
        print(f"Error retrieving stock data: {e}")