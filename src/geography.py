"""World geography for PriceVision.

Maps a broad set of countries across every continent to their major cities,
each tagged with a price *tier* (1 = most expensive megacity … 5 = small town).
A house price is driven by:

    market_factor = country_factor * TIER_MULT[city_tier]

Literally enumerating every settlement on Earth is not feasible, so the model
is designed to **generalise to any city**: the transferable signal lives in
``city_tier`` (a numeric feature), while the specific ``city`` name is an
optional high-cardinality hint. A city that is not in this table can still be
priced — pass its tier (or let it default) and the model uses the country +
tier signal.
"""
from __future__ import annotations

# Multiplier applied on top of the country factor, per city price tier.
TIER_MULT = {1: 1.70, 2: 1.30, 3: 1.00, 4: 0.80, 5: 0.62}
DEFAULT_TIER = 3

# country -> (country_factor_in_USD_terms, {city: tier})
# country_factor is normalised so that Bangladesh ~ 1.0.
COUNTRIES: dict[str, tuple[float, dict[str, int]]] = {
    # ---------------- South Asia ----------------
    "Bangladesh": (1.0, {
        "Dhaka": 1, "Chittagong": 2, "Sylhet": 3, "Khulna": 3, "Rajshahi": 4,
        "Barisal": 4, "Rangpur": 4, "Mymensingh": 4, "Comilla": 4, "Cox's Bazar": 3,
        "Narayanganj": 3, "Gazipur": 3,
    }),
    "India": (1.15, {
        "Mumbai": 1, "Delhi": 1, "Bangalore": 1, "Hyderabad": 2, "Chennai": 2,
        "Pune": 2, "Kolkata": 2, "Ahmedabad": 3, "Gurgaon": 1, "Noida": 2,
        "Jaipur": 3, "Lucknow": 3, "Surat": 3, "Kochi": 3, "Indore": 4,
        "Nagpur": 4, "Bhopal": 4, "Patna": 4,
    }),
    "Pakistan": (0.9, {
        "Karachi": 1, "Lahore": 2, "Islamabad": 1, "Rawalpindi": 3,
        "Faisalabad": 4, "Multan": 4, "Peshawar": 4, "Quetta": 4, "Sialkot": 4,
    }),
    "Sri Lanka": (1.1, {
        "Colombo": 1, "Kandy": 3, "Galle": 3, "Jaffna": 4, "Negombo": 3,
    }),
    "Nepal": (0.85, {
        "Kathmandu": 1, "Pokhara": 3, "Lalitpur": 2, "Biratnagar": 4,
    }),
    # ---------------- East & Southeast Asia ----------------
    "China": (3.6, {
        "Shanghai": 1, "Beijing": 1, "Shenzhen": 1, "Guangzhou": 2,
        "Hangzhou": 2, "Chengdu": 3, "Wuhan": 3, "Xi'an": 3, "Chongqing": 3,
        "Nanjing": 2, "Tianjin": 3, "Suzhou": 2,
    }),
    "Japan": (5.4, {
        "Tokyo": 1, "Osaka": 2, "Yokohama": 2, "Nagoya": 3, "Kyoto": 2,
        "Fukuoka": 3, "Sapporo": 3, "Kobe": 3, "Sendai": 4,
    }),
    "South Korea": (4.6, {
        "Seoul": 1, "Busan": 2, "Incheon": 3, "Daegu": 3, "Daejeon": 4,
        "Gwangju": 4, "Suwon": 2,
    }),
    "Singapore": (7.5, {"Singapore": 1}),
    "Malaysia": (2.1, {
        "Kuala Lumpur": 1, "Penang": 2, "Johor Bahru": 3, "Ipoh": 4,
        "Kuching": 4, "Kota Kinabalu": 4,
    }),
    "Thailand": (2.0, {
        "Bangkok": 1, "Phuket": 2, "Chiang Mai": 3, "Pattaya": 3,
        "Nonthaburi": 3, "Krabi": 4,
    }),
    "Indonesia": (1.8, {
        "Jakarta": 1, "Surabaya": 3, "Bandung": 3, "Bali": 2, "Medan": 4,
        "Semarang": 4, "Makassar": 4,
    }),
    "Vietnam": (1.5, {
        "Ho Chi Minh City": 1, "Hanoi": 2, "Da Nang": 3, "Hai Phong": 4,
        "Can Tho": 4,
    }),
    "Philippines": (1.7, {
        "Manila": 1, "Makati": 1, "Cebu": 3, "Davao": 4, "Quezon City": 2,
    }),
    # ---------------- Middle East ----------------
    "UAE": (3.3, {
        "Dubai": 1, "Abu Dhabi": 1, "Sharjah": 3, "Ajman": 4, "Al Ain": 4,
        "Ras Al Khaimah": 4,
    }),
    "Saudi Arabia": (2.6, {
        "Riyadh": 1, "Jeddah": 2, "Mecca": 2, "Medina": 3, "Dammam": 3,
        "Khobar": 3,
    }),
    "Qatar": (3.8, {"Doha": 1, "Al Rayyan": 3, "Al Wakrah": 4}),
    "Israel": (5.0, {
        "Tel Aviv": 1, "Jerusalem": 2, "Haifa": 3, "Netanya": 3, "Eilat": 3,
    }),
    "Turkey": (2.2, {
        "Istanbul": 1, "Ankara": 2, "Izmir": 2, "Antalya": 3, "Bursa": 4,
        "Adana": 4,
    }),
    # ---------------- Europe ----------------
    "UK": (5.6, {
        "London": 1, "Manchester": 2, "Birmingham": 2, "Edinburgh": 2,
        "Bristol": 2, "Glasgow": 3, "Leeds": 3, "Liverpool": 3, "Cambridge": 1,
        "Oxford": 1, "Cardiff": 3, "Sheffield": 4, "Belfast": 4,
    }),
    "Ireland": (5.2, {"Dublin": 1, "Cork": 3, "Galway": 3, "Limerick": 4}),
    "France": (4.9, {
        "Paris": 1, "Nice": 2, "Lyon": 2, "Marseille": 3, "Bordeaux": 3,
        "Toulouse": 3, "Lille": 3, "Nantes": 4, "Strasbourg": 4,
    }),
    "Germany": (4.7, {
        "Munich": 1, "Frankfurt": 1, "Berlin": 2, "Hamburg": 2, "Stuttgart": 2,
        "Cologne": 3, "Dusseldorf": 3, "Dresden": 4, "Leipzig": 4,
    }),
    "Italy": (4.2, {
        "Milan": 1, "Rome": 1, "Florence": 2, "Venice": 2, "Turin": 3,
        "Naples": 3, "Bologna": 3, "Genoa": 4, "Palermo": 4,
    }),
    "Spain": (3.8, {
        "Madrid": 1, "Barcelona": 1, "Valencia": 3, "Seville": 3, "Bilbao": 3,
        "Malaga": 3, "Zaragoza": 4,
    }),
    "Portugal": (3.4, {
        "Lisbon": 1, "Porto": 2, "Faro": 3, "Braga": 4, "Coimbra": 4,
    }),
    "Netherlands": (5.0, {
        "Amsterdam": 1, "Rotterdam": 2, "The Hague": 2, "Utrecht": 2,
        "Eindhoven": 3, "Groningen": 4,
    }),
    "Switzerland": (7.2, {
        "Zurich": 1, "Geneva": 1, "Basel": 2, "Bern": 2, "Lausanne": 2,
        "Lugano": 3,
    }),
    "Sweden": (4.8, {"Stockholm": 1, "Gothenburg": 2, "Malmo": 3, "Uppsala": 3}),
    "Norway": (5.5, {"Oslo": 1, "Bergen": 2, "Trondheim": 3, "Stavanger": 3}),
    "Poland": (2.4, {
        "Warsaw": 1, "Krakow": 2, "Wroclaw": 3, "Gdansk": 3, "Poznan": 4,
        "Lodz": 4,
    }),
    "Greece": (2.8, {
        "Athens": 1, "Thessaloniki": 3, "Patras": 4, "Heraklion": 4,
    }),
    "Russia": (2.5, {
        "Moscow": 1, "Saint Petersburg": 2, "Novosibirsk": 4, "Kazan": 4,
        "Yekaterinburg": 4, "Sochi": 3,
    }),
    # ---------------- North America ----------------
    "USA": (6.2, {
        "New York": 1, "San Francisco": 1, "Los Angeles": 1, "Seattle": 1,
        "Boston": 1, "Washington DC": 1, "San Diego": 2, "Miami": 2,
        "Chicago": 2, "Austin": 2, "Denver": 2, "Portland": 2, "Dallas": 3,
        "Houston": 3, "Atlanta": 3, "Phoenix": 3, "Nashville": 3, "Charlotte": 3,
        "Detroit": 4, "Cleveland": 4, "Memphis": 4, "Kansas City": 4,
    }),
    "Canada": (4.9, {
        "Toronto": 1, "Vancouver": 1, "Montreal": 2, "Ottawa": 2, "Calgary": 3,
        "Edmonton": 3, "Quebec City": 3, "Winnipeg": 4, "Halifax": 4,
    }),
    "Mexico": (2.0, {
        "Mexico City": 1, "Monterrey": 2, "Guadalajara": 2, "Cancun": 3,
        "Tijuana": 3, "Puebla": 4, "Merida": 4,
    }),
    # ---------------- South America ----------------
    "Brazil": (2.1, {
        "Sao Paulo": 1, "Rio de Janeiro": 1, "Brasilia": 2, "Curitiba": 3,
        "Belo Horizonte": 3, "Porto Alegre": 3, "Salvador": 4, "Fortaleza": 4,
    }),
    "Argentina": (1.9, {
        "Buenos Aires": 1, "Cordoba": 3, "Rosario": 3, "Mendoza": 4,
    }),
    "Chile": (2.3, {"Santiago": 1, "Valparaiso": 3, "Concepcion": 4}),
    "Colombia": (1.7, {
        "Bogota": 1, "Medellin": 2, "Cali": 3, "Cartagena": 3, "Barranquilla": 4,
    }),
    "Peru": (1.6, {"Lima": 1, "Arequipa": 3, "Cusco": 3, "Trujillo": 4}),
    # ---------------- Africa ----------------
    "Egypt": (1.4, {
        "Cairo": 1, "Giza": 2, "Alexandria": 3, "Sharm El Sheikh": 3,
        "Luxor": 4, "Aswan": 4,
    }),
    "Nigeria": (1.5, {
        "Lagos": 1, "Abuja": 2, "Port Harcourt": 3, "Kano": 4, "Ibadan": 4,
    }),
    "South Africa": (2.2, {
        "Cape Town": 1, "Johannesburg": 1, "Durban": 3, "Pretoria": 3,
        "Port Elizabeth": 4,
    }),
    "Kenya": (1.5, {"Nairobi": 1, "Mombasa": 3, "Kisumu": 4, "Nakuru": 4}),
    "Morocco": (1.6, {
        "Casablanca": 1, "Marrakesh": 2, "Rabat": 2, "Tangier": 3, "Fes": 4,
    }),
    # ---------------- Oceania ----------------
    "Australia": (5.3, {
        "Sydney": 1, "Melbourne": 1, "Brisbane": 2, "Perth": 2, "Canberra": 2,
        "Adelaide": 3, "Gold Coast": 3, "Hobart": 4, "Darwin": 4,
    }),
    "New Zealand": (4.4, {
        "Auckland": 1, "Wellington": 2, "Christchurch": 3, "Queenstown": 2,
        "Hamilton": 4,
    }),
}

# Convenience views ---------------------------------------------------------
COUNTRY_LEVELS = sorted(COUNTRIES.keys())
COUNTRY_FACTOR = {c: COUNTRIES[c][0] for c in COUNTRIES}


def cities_for(country: str) -> list[str]:
    """Return the known cities for a country (sorted)."""
    if country not in COUNTRIES:
        return []
    return sorted(COUNTRIES[country][1].keys())


def all_cities() -> list[tuple[str, str, int]]:
    """Yield (country, city, tier) for every known city."""
    out = []
    for country, (_factor, cities) in COUNTRIES.items():
        for city, tier in cities.items():
            out.append((country, city, tier))
    return out


def tier_of(country: str, city: str | None = None) -> int:
    """Price tier for a (country, city). Falls back to DEFAULT_TIER."""
    if country in COUNTRIES and city:
        return COUNTRIES[country][1].get(city, DEFAULT_TIER)
    return DEFAULT_TIER


def factor_of(country: str, city: str | None = None) -> float:
    """Combined market factor = country_factor * tier multiplier."""
    country_factor = COUNTRY_FACTOR.get(country, 1.0)
    return country_factor * TIER_MULT[tier_of(country, city)]


def n_countries() -> int:
    return len(COUNTRIES)


def n_cities() -> int:
    return sum(len(c[1]) for c in COUNTRIES.values())
