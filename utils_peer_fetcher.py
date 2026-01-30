"""
ULTRA-FAST Industry Peer Fetcher - PRECISE HTML PARSING
=================================
Supports BOTH NSE (.NS) and BSE (.BO) exchanges
NO HARDCODED PEERS
"""
import yfinance as yf
import logging
import requests
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


def get_peers_from_yahoo_comparison(ticker: str, max_peers: int = 20, exclude_self: bool = True):
    """
    Scrapes Yahoo Finance comparison carousel
    Automatically detects exchange (.NS or .BO) from ticker
    
    Returns: List of peer tickers (without exchange suffix)
    """
    ticker_upper = ticker.upper()
    
    # Detect exchange from ticker
    if '.BO' in ticker_upper:
        exchange_suffix = '.BO'
        ticker_base = ticker_upper.replace('.BO', '')
    elif '.NS' in ticker_upper:
        exchange_suffix = '.NS'
        ticker_base = ticker_upper.replace('.NS', '')
    else:
        # Default to .NS
        exchange_suffix = '.NS'
        ticker_base = ticker_upper
    
    full_ticker = f"{ticker_base}{exchange_suffix}"
    peers = []
    
    try:
        url = f"https://finance.yahoo.com/quote/{full_ticker}/comparison"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        print(f"[PeerFetcher] Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            print(f"[PeerFetcher] HTTP {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # STRATEGY 1: Target carousel links - BOTH .NS and .BO
        print("[PeerFetcher] Strategy 1: Carousel links...")
        carousel_links = soup.find_all('a', class_='loud-link')
        
        for link in carousel_links:
            href = link.get('href', '')
            if '/quote/' in href and ('.NS' in href or '.BO' in href):
                match = re.search(r'/quote/([A-Z0-9&\-]+)\.(NS|BO)', href)
                if match:
                    peer = match.group(1)
                    if exclude_self and peer == ticker_base:
                        continue
                    if peer not in peers:
                        peers.append(peer)
                        print(f"   Found: {peer}")
        
        # STRATEGY 2: Scan spans - BOTH .NS and .BO
        if len(peers) < 3:
            print("[PeerFetcher] Strategy 2: Span elements...")
            for span in soup.find_all('span'):
                text = span.get_text(strip=True)
                if re.match(r'^[A-Z][A-Z0-9&\-]*\.(NS|BO)$', text):
                    peer = text.replace('.NS', '').replace('.BO', '')
                    if exclude_self and peer == ticker_base:
                        continue
                    if peer not in peers:
                        peers.append(peer)
                        print(f"   Found: {peer}")
        
        # STRATEGY 3: Full text scan - BOTH .NS and .BO
        if len(peers) < 3:
            print("[PeerFetcher] Strategy 3: Full text scan...")
            all_text = soup.get_text()
            pattern = r'\b([A-Z][A-Z0-9&\-]{2,})\.(NS|BO)\b'
            matches = re.findall(pattern, all_text)
            for peer, _ in matches:
                if exclude_self and peer == ticker_base:
                    continue
                if peer not in peers:
                    peers.append(peer)
                    print(f"   Found: {peer}")
        
        # STRATEGY 4: Data attributes
        if len(peers) < 3:
            print("[PeerFetcher] Strategy 4: Data attributes...")
            for elem in soup.find_all(attrs={'data-symbol': True}):
                symbol = elem.get('data-symbol', '')
                if '.NS' in symbol or '.BO' in symbol:
                    peer = symbol.replace('.NS', '').replace('.BO', '')
                    if exclude_self and peer == ticker_base:
                        continue
                    if peer not in peers:
                        peers.append(peer)
                        print(f"   Found: {peer}")
        
        # STRATEGY 5: ARIA labels
        if len(peers) < 3:
            print("[PeerFetcher] Strategy 5: ARIA labels...")
            for elem in soup.find_all(attrs={'aria-label': True}):
                label = elem.get('aria-label', '')
                if '.NS' in label or '.BO' in label:
                    match = re.search(r'([A-Z][A-Z0-9&\-]+)\.(NS|BO)', label)
                    if match:
                        peer = match.group(1)
                        if exclude_self and peer == ticker_base:
                            continue
                        if peer not in peers:
                            peers.append(peer)
                            print(f"   Found: {peer}")
        
        # Remove duplicates
        unique_peers = []
        seen = set()
        for peer in peers:
            clean = peer.replace('.BO', '').replace('.NS', '').strip()
            if clean and clean not in seen:
                seen.add(clean)
                unique_peers.append(clean)
        
        result = unique_peers[:max_peers]
        
        if result:
            print(f"\n[PeerFetcher] SUCCESS: Found {len(result)} peers for {full_ticker}")
            print(f"Peers: {', '.join(result)}")
            return result
        else:
            print(f"\n[PeerFetcher] No peers found")
            return []
        
    except Exception as e:
        print(f"[PeerFetcher] Error: {e}")
        return []


def get_industry_peers(ticker: str, max_peers: int = 20, exclude_self: bool = True):
    """Main function - Pure web scraping, no hardcoded data"""
    ticker_upper = ticker.upper()
    
    print(f"\n{'='*70}")
    print(f"[PeerFetcher] Finding peers for {ticker_upper}")
    print('='*70)
    
    peers = get_peers_from_yahoo_comparison(ticker_upper, max_peers, exclude_self)
    
    if peers:
        print(f"\n{'='*70}")
        print(f"[PeerFetcher] FINAL RESULT: {len(peers)} peers found")
        print(f"\nPeer List:")
        for i, peer in enumerate(peers, 1):
            print(f"   {i:2d}. {peer}")
        print('='*70)
        return peers
    else:
        print(f"\n{'='*70}")
        print(f"[PeerFetcher] FAILED: No peers found for {ticker_upper}")
        print("="*70)
        return []


# Alias for compatibility
get_industry_peers_fast = get_industry_peers


if __name__ == "__main__":
    # Test
    test_tickers = ["TATASTEEL", "ADVAIT.BO", "RELIANCE.NS"]
    
    for ticker in test_tickers:
        peers = get_industry_peers(ticker, max_peers=15)
        print(f"\n{ticker}: {len(peers)} peers\n")
