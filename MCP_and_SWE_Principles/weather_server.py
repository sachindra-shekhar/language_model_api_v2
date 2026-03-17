
import json
import httpx
from mcp.server.fastmcp import FastMCP

# ── Initialize the MCP Server ────────────────────────────────
mcp = FastMCP("weather_mcp")

import truststore
truststore.inject_into_ssl()


@mcp.tool()
async def get_weather(latitude: float, longitude: float) -> str:
    """Get the current weather for a location.

    Args:
        latitude: Latitude of the location (e.g., 48.8566 for Paris)
        longitude: Longitude of the location (e.g., 2.3522 for Paris)

    Returns:
        A JSON string with temperature, wind speed, and conditions.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": latitude, "longitude": longitude, "current_weather": True}

    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    weather = data["current_weather"]
    return json.dumps({
        "temperature_celsius": weather["temperature"],
        "windspeed_kmh": weather["windspeed"],
        "weather_code": weather["weathercode"],
        "is_day": weather["is_day"] == 1
    }, indent=2)


@mcp.tool()
async def get_weather_comparison(
    city1_lat: float, city1_lon: float, city1_name: str,
    city2_lat: float, city2_lon: float, city2_name: str
) -> str:
    """Compare current weather between two cities.

    Args:
        city1_lat: Latitude of the first city
        city1_lon: Longitude of the first city
        city1_name: Name of the first city
        city2_lat: Latitude of the second city
        city2_lon: Longitude of the second city
        city2_name: Name of the second city
    """
    async with httpx.AsyncClient() as http_client:
        cities = {}
        for name, lat, lon in [(city1_name, city1_lat, city1_lon),
                                (city2_name, city2_lat, city2_lon)]:
            resp = await http_client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon, "current_weather": True}
            )
            resp.raise_for_status()
            w = resp.json()["current_weather"]
            cities[name] = {"temperature_celsius": w["temperature"], "windspeed_kmh": w["windspeed"]}

    return json.dumps({
        "comparison": cities,
        "warmer_city": max(cities, key=lambda c: cities[c]["temperature_celsius"]),
        "windier_city": max(cities, key=lambda c: cities[c]["windspeed_kmh"]),
    }, indent=2)


# ── Define a Resource (read-only data endpoint) ─────────────
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

@mcp.resource("weather://codes")
def get_weather_codes() -> str:
    """A lookup table of WMO weather interpretation codes."""
    return json.dumps(WEATHER_CODES, indent=2)


if __name__ == "__main__":
    mcp.run()  # Uses stdio transport by default
