// Market iBoard JavaScript

let marketData = [];
let filteredData = [];
let currentGroup = 'VN30'; // Default to VN30

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupNavigation();
    setupSearch();
    loadMarketIndices();
    loadMarketData(currentGroup);
    setInterval(() => {
        loadMarketIndices();
        loadMarketData(currentGroup);
    }, 30000); // Refresh every 30 seconds
});

function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Allow home link to navigate normally
            if (this.getAttribute('href') === '/') {
                return;
            }
            
            e.preventDefault();
            const text = this.textContent.trim();
            
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // Determine group from link text
            if (text === 'VN30') currentGroup = 'VN30';
            else if (text === 'HNX30') currentGroup = 'HNX30';
            else if (text === 'HOSE') currentGroup = 'HOSE';
            else if (text === 'HNX') currentGroup = 'HNX';
            else if (text === 'UPCOM') currentGroup = 'UPCOM';
            
            loadMarketData(currentGroup);
        });
    });
}

function setupSearch() {
    const searchInput = document.getElementById('searchInput');
    searchInput.addEventListener('input', function(e) {
        const term = e.target.value.toLowerCase().trim();
        if (term) {
            filteredData = marketData.filter(stock => 
                stock.symbol.toLowerCase().includes(term)
            );
        } else {
            filteredData = [...marketData];
        }
        renderTable();
    });
}

async function loadMarketIndices() {
    try {
        const response = await fetch('/api/market/indices');
        if (!response.ok) return;
        
        const data = await response.json();
        const indices = data.indices;
        
        // Update VN-INDEX
        if (indices.VNINDEX) {
            const vnindex = indices.VNINDEX;
            document.getElementById('vnindex').textContent = vnindex.value.toFixed(2);
            const vnChange = document.getElementById('vnindex-change');
            const changeClass = vnindex.change >= 0 ? 'up' : 'down';
            vnChange.textContent = `${vnindex.change >= 0 ? '+' : ''}${vnindex.change.toFixed(2)} (${vnindex.change >= 0 ? '+' : ''}${vnindex.percent.toFixed(2)}%)`;
            vnChange.className = `stat-change ${changeClass}`;
            document.getElementById('vnindex').className = `stat-value ${changeClass}`;
        }
        
        // Update HNX
        if (indices.HNX) {
            const hnx = indices.HNX;
            document.getElementById('hnx').textContent = hnx.value.toFixed(2);
            const hnxChange = document.getElementById('hnx-change');
            const changeClass = hnx.change >= 0 ? 'up' : 'down';
            hnxChange.textContent = `${hnx.change >= 0 ? '+' : ''}${hnx.change.toFixed(2)} (${hnx.change >= 0 ? '+' : ''}${hnx.percent.toFixed(2)}%)`;
            hnxChange.className = `stat-change ${changeClass}`;
            document.getElementById('hnx').className = `stat-value ${changeClass}`;
        }
        
        // Update VN30
        if (indices.VN30) {
            const vn30 = indices.VN30;
            const vn30El = document.getElementById('vn30');
            const vn30ChangeEl = document.getElementById('vn30-change');
            if (vn30El && vn30ChangeEl) {
                vn30El.textContent = vn30.value.toFixed(2);
                const changeClass = vn30.change >= 0 ? 'up' : (vn30.change === 0 ? 'ref' : 'down');
                vn30ChangeEl.textContent = `${vn30.change >= 0 ? '+' : ''}${vn30.change.toFixed(2)} (${vn30.change >= 0 ? '+' : ''}${vn30.percent.toFixed(2)}%)`;
                vn30ChangeEl.className = `stat-change ${changeClass}`;
                vn30El.className = `stat-value ${changeClass}`;
            }
        }
    } catch (error) {
        console.error('Error loading indices:', error);
    }
}

async function loadMarketData(group = 'VN30') {
    try {
        const url = `/api/market/data${group ? '?group=' + group : ''}`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data.data || data.data.length === 0) {
            showError('Không có dữ liệu');
            return;
        }
        
        console.log(`Loaded ${data.data.length} stocks for ${group || 'all'}`);
        
        marketData = processData(data.data);
        filteredData = [...marketData];
        
        updateStats();
        renderTable();
        
    } catch (error) {
        console.error('Error loading data:', error);
        showError('Lỗi tải dữ liệu: ' + error.message);
    }
}

function processData(rawData) {
    return rawData.map(stock => {
        const refPrice = parseFloat(stock.reference_price || 0);
        const closePrice = parseFloat(stock.close_price || 0);
        const change = parseFloat(stock.price_change || 0);
        const percentChange = parseFloat(stock.percent_change || 0);
        const exchange = stock.exchange || 'HOSE';
        
        // Determine price limits based on exchange
        let ceilingPercent, floorPercent;
        if (exchange === 'HNX') {
            ceilingPercent = 10;
            floorPercent = -10;
        } else if (exchange === 'UPCOM') {
            ceilingPercent = 15;
            floorPercent = -15;
        } else { // HOSE
            ceilingPercent = 7;
            floorPercent = -7;
        }
        
        return {
            symbol: stock.symbol || '',
            exchange: exchange,
            // Price levels
            ceiling: parseFloat(stock.ceiling_price || 0),
            floor: parseFloat(stock.floor_price || 0),
            reference: refPrice,
            // Bid levels (buy orders)
            bid1Price: parseFloat(stock.bid_price_1 || 0),
            bid1Vol: parseInt(stock.bid_vol_1 || 0),
            bid2Price: parseFloat(stock.bid_price_2 || 0),
            bid2Vol: parseInt(stock.bid_vol_2 || 0),
            bid3Price: parseFloat(stock.bid_price_3 || 0),
            bid3Vol: parseInt(stock.bid_vol_3 || 0),
            // Match (current price)
            price: closePrice,
            volume: parseInt(stock.total_trades || 0),
            // Change
            change: change,
            percentChange: percentChange,
            // Ask levels (sell orders)
            ask1Price: parseFloat(stock.ask_price_1 || 0),
            ask1Vol: parseInt(stock.ask_vol_1 || 0),
            ask2Price: parseFloat(stock.ask_price_2 || 0),
            ask2Vol: parseInt(stock.ask_vol_2 || 0),
            ask3Price: parseFloat(stock.ask_price_3 || 0),
            ask3Vol: parseInt(stock.ask_vol_3 || 0),
            // Additional info
            totalVolume: parseInt(stock.total_trades || 0),
            high: parseFloat(stock.high_price || 0),
            low: parseFloat(stock.low_price || 0),
            foreignBuy: parseInt(stock.foreign_buy_volume || 0),
            foreignSell: Math.abs(parseInt(stock.foreign_sell_volume || 0)),
            room: parseInt(stock.foreign_remain_room || stock.foreign_room || 0),
            // Price limits for coloring
            ceilingPercent: ceilingPercent,
            floorPercent: floorPercent
        };
    }).filter(s => s.symbol);
}

function updateStats() {
    const gainers = marketData.filter(s => s.change > 0).length;
    const losers = marketData.filter(s => s.change < 0).length;
    const unchanged = marketData.filter(s => s.change === 0).length;
    
    document.getElementById('gain-count').textContent = gainers;
    document.getElementById('loss-count').textContent = losers;
    document.getElementById('ref-count').textContent = unchanged;
}

function renderTable() {
    const tbody = document.getElementById('priceTableBody');
    
    if (filteredData.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="24" class="loading-cell">
                    <i class="fas fa-inbox"></i>
                    Không tìm thấy dữ liệu
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = filteredData.map(stock => {
        const priceClass = getPriceClassByPercent(stock.price, stock.reference, stock.ceilingPercent, stock.floorPercent);
        const changeClass = stock.change >= 0 ? 'up' : 'down';
        const symbolClass = getPriceClassByPercent(stock.price, stock.reference, stock.ceilingPercent, stock.floorPercent);
        
        // Calculate color classes for bid/ask prices
        const bid3Class = getPriceClassByPercent(stock.bid3Price, stock.reference, stock.ceilingPercent, stock.floorPercent);
        const bid2Class = getPriceClassByPercent(stock.bid2Price, stock.reference, stock.ceilingPercent, stock.floorPercent);
        const bid1Class = getPriceClassByPercent(stock.bid1Price, stock.reference, stock.ceilingPercent, stock.floorPercent);
        const ask1Class = getPriceClassByPercent(stock.ask1Price, stock.reference, stock.ceilingPercent, stock.floorPercent);
        const ask2Class = getPriceClassByPercent(stock.ask2Price, stock.reference, stock.ceilingPercent, stock.floorPercent);
        const ask3Class = getPriceClassByPercent(stock.ask3Price, stock.reference, stock.ceilingPercent, stock.floorPercent);
        
        return `
            <tr>
                <td class="td-symbol ${symbolClass}">${stock.symbol}</td>
                <td class="td-ceil num">${fmt(stock.ceiling)}</td>
                <td class="td-floor num">${fmt(stock.floor)}</td>
                <td class="td-ref num">${fmt(stock.reference)}</td>
                
                <!-- Bid (Buy) Side -->
                <td class="td-bid td-bid-price ${bid3Class} num">${fmt(stock.bid3Price)}</td>
                <td class="td-bid td-bid-vol num">${fmtVol(stock.bid3Vol)}</td>
                <td class="td-bid td-bid-price ${bid2Class} num">${fmt(stock.bid2Price)}</td>
                <td class="td-bid td-bid-vol num">${fmtVol(stock.bid2Vol)}</td>
                <td class="td-bid td-bid-price ${bid1Class} num">${fmt(stock.bid1Price)}</td>
                <td class="td-bid td-bid-vol num">${fmtVol(stock.bid1Vol)}</td>
                
                <!-- Match Price -->
                <td class="td-match-price ${priceClass} num">${fmt(stock.price)}</td>
                <td class="td-match-vol num">${fmtVol(stock.volume)}</td>
                
                <!-- Change -->
                <td class="td-change ${changeClass} num">${fmtChange(stock.change)}</td>
                <td class="td-percent ${changeClass} num">${fmtPercent(stock.percentChange)}</td>
                
                <!-- Ask (Sell) Side -->
                <td class="td-ask td-ask-price ${ask1Class} num">${fmt(stock.ask1Price)}</td>
                <td class="td-ask td-ask-vol num">${fmtVol(stock.ask1Vol)}</td>
                <td class="td-ask td-ask-price ${ask2Class} num">${fmt(stock.ask2Price)}</td>
                <td class="td-ask td-ask-vol num">${fmtVol(stock.ask2Vol)}</td>
                <td class="td-ask td-ask-price ${ask3Class} num">${fmt(stock.ask3Price)}</td>
                <td class="td-ask td-ask-vol num">${fmtVol(stock.ask3Vol)}</td>
                
                <!-- Additional Info -->
                <td class="td-vol num">${fmtVol(stock.totalVolume)}</td>
                <td class="td-high num">${fmt(stock.high)}</td>
                <td class="td-low num">${fmt(stock.low)}</td>
                <td class="td-foreign num">${fmtVol(stock.foreignBuy)}</td>
                <td class="td-foreign num">${fmtVol(stock.foreignSell)}</td>
            </tr>
        `;
    }).join('');
}

function getPriceClass(price, ref, ceiling, floor) {
    if (!price || !ref) return 'ref';
    if (price >= ceiling) return 'ceil-price';
    if (price <= floor) return 'floor-price';
    if (price > ref) return 'up';
    if (price < ref) return 'down';
    return 'ref';
}

function getPriceClassByPercent(price, ref, ceilingPercent, floorPercent) {
    if (!price || !ref) return 'ref';
    
    const percentChange = ((price - ref) / ref) * 100;
    
    // Trần (ceiling)
    if (percentChange >= ceilingPercent) return 'ceil-price';
    // Sàn (floor)
    if (percentChange <= floorPercent) return 'floor-price';
    // Tăng
    if (percentChange > 0) return 'up';
    // Giảm
    if (percentChange < 0) return 'down';
    // Tham chiếu
    return 'ref';
}

// Formatting functions
function fmt(value) {
    if (!value || value === 0) return '-';
    return value.toFixed(2);
}

function fmtVol(value) {
    if (!value || value === 0) return '-';
    if (value >= 1000000) {
        return (value / 1000000).toFixed(1) + 'M';
    }
    if (value >= 1000) {
        return (value / 1000).toFixed(1) + 'K';
    }
    return value.toLocaleString('en-US');
}

function fmtChange(value) {
    if (!value || value === 0) return '0.00';
    return (value >= 0 ? '+' : '') + value.toFixed(2);
}

function fmtPercent(value) {
    if (!value || value === 0) return '0.00%';
    return (value >= 0 ? '+' : '') + value.toFixed(2) + '%';
}

function showError(message) {
    const tbody = document.getElementById('priceTableBody');
    tbody.innerHTML = `
        <tr>
            <td colspan="24" class="loading-cell" style="color: #ef4444;">
                <i class="fas fa-exclamation-triangle"></i>
                ${message}
            </td>
        </tr>
    `;
}
