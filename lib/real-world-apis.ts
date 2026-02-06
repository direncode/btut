/**
 * Real-World API Integration Layer
 * All APIs are free and require ZERO user configuration.
 *
 * APIs used:
 * - Open-Meteo: Weather data (no key)
 * - Overpass/OSM: Road network data (no key)
 * - NASA: Mars rover telemetry (public DEMO_KEY)
 * - OpenSky Network: Live flight tracking (no key)
 * - USGS: Earthquake data (no key)
 * - World Bank: Economic/population data (no key)
 */

// ============================================================================
// Open-Meteo Weather API (Traffic Domain)
// https://open-meteo.com — completely free, no API key
// ============================================================================

export interface WeatherData {
  latitude: number;
  longitude: number;
  temperature_c: number;
  windspeed_kmh: number;
  winddirection: number;
  precipitation_mm: number;
  weathercode: number;
  weather_description: string;
  visibility_km: number;
  humidity_percent: number;
  is_day: boolean;
  traffic_impact_score: number; // 0-1 computed score
  timestamp: string;
}

const WEATHER_CODES: Record<number, string> = {
  0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
  45: 'Fog', 48: 'Depositing rime fog',
  51: 'Light drizzle', 53: 'Moderate drizzle', 55: 'Dense drizzle',
  61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
  71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow',
  77: 'Snow grains', 80: 'Slight showers', 81: 'Moderate showers', 82: 'Violent showers',
  85: 'Slight snow showers', 86: 'Heavy snow showers',
  95: 'Thunderstorm', 96: 'Thunderstorm w/ slight hail', 99: 'Thunderstorm w/ heavy hail',
};

function computeTrafficImpact(weathercode: number, precipitation: number, windspeed: number, visibility: number): number {
  let impact = 0;
  if (weathercode >= 95) impact += 0.4;
  else if (weathercode >= 71) impact += 0.35;
  else if (weathercode >= 61) impact += 0.2;
  else if (weathercode >= 45) impact += 0.25;
  impact += Math.min(0.3, precipitation / 20);
  impact += Math.min(0.2, windspeed / 100);
  if (visibility < 10) impact += Math.min(0.2, (10 - visibility) / 10 * 0.2);
  return Math.min(1, impact);
}

export async function fetchWeather(lat: number = 40.7128, lon: number = -74.006): Promise<WeatherData> {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,wind_direction_10m,is_day,visibility`;
  const res = await fetch(url, { next: { revalidate: 300 } });
  if (!res.ok) throw new Error(`Open-Meteo API error: ${res.status}`);
  const data = await res.json();
  const c = data.current;
  const visibility = (c.visibility ?? 10000) / 1000;

  return {
    latitude: data.latitude,
    longitude: data.longitude,
    temperature_c: c.temperature_2m,
    windspeed_kmh: c.wind_speed_10m,
    winddirection: c.wind_direction_10m,
    precipitation_mm: c.precipitation,
    weathercode: c.weather_code,
    weather_description: WEATHER_CODES[c.weather_code] ?? 'Unknown',
    visibility_km: visibility,
    humidity_percent: c.relative_humidity_2m,
    is_day: c.is_day === 1,
    traffic_impact_score: computeTrafficImpact(c.weather_code, c.precipitation, c.wind_speed_10m, visibility),
    timestamp: c.time,
  };
}

// ============================================================================
// Overpass / OpenStreetMap Road Network (Traffic Domain)
// https://overpass-api.de — completely free, no API key
// ============================================================================

export interface RoadSegment {
  id: number;
  type: string;
  name: string | null;
  lanes: number;
  maxspeed: string | null;
  nodes: Array<{ lat: number; lon: number }>;
}

export interface RoadNetwork {
  road_count: number;
  intersection_count: number;
  total_length_km: number;
  roads: RoadSegment[];
  bounds: { south: number; west: number; north: number; east: number };
  timestamp: string;
}

export async function fetchRoadNetwork(
  lat: number = 40.7128, lon: number = -74.006, radius: number = 500
): Promise<RoadNetwork> {
  const query = `[out:json][timeout:25];way(around:${radius},${lat},${lon})["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential)$"];out body;>;out skel qt;`;
  const url = `https://overpass-api.de/api/interpreter?data=${encodeURIComponent(query)}`;
  const res = await fetch(url, { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error(`Overpass API error: ${res.status}`);
  const data = await res.json();

  const nodeMap = new Map<number, { lat: number; lon: number }>();
  const roads: RoadSegment[] = [];

  for (const el of data.elements) {
    if (el.type === 'node') {
      nodeMap.set(el.id, { lat: el.lat, lon: el.lon });
    }
  }

  for (const el of data.elements) {
    if (el.type === 'way' && el.tags?.highway) {
      const nodes = (el.nodes ?? []).map((nid: number) => nodeMap.get(nid)).filter(Boolean) as Array<{ lat: number; lon: number }>;
      roads.push({
        id: el.id,
        type: el.tags.highway,
        name: el.tags.name ?? null,
        lanes: parseInt(el.tags.lanes ?? '2', 10),
        maxspeed: el.tags.maxspeed ?? null,
        nodes,
      });
    }
  }

  let totalLength = 0;
  for (const road of roads) {
    for (let i = 1; i < road.nodes.length; i++) {
      const a = road.nodes[i - 1], b = road.nodes[i];
      totalLength += haversine(a.lat, a.lon, b.lat, b.lon);
    }
  }

  const intersectionNodeIds = new Set<number>();
  const nodeCounts = new Map<number, number>();
  for (const el of data.elements) {
    if (el.type === 'way' && el.nodes) {
      for (const nid of el.nodes) {
        nodeCounts.set(nid, (nodeCounts.get(nid) ?? 0) + 1);
      }
    }
  }
  for (const [nid, count] of nodeCounts) {
    if (count >= 2) intersectionNodeIds.add(nid);
  }

  return {
    road_count: roads.length,
    intersection_count: intersectionNodeIds.size,
    total_length_km: totalLength,
    roads: roads.slice(0, 200),
    bounds: {
      south: lat - radius / 111320,
      west: lon - radius / (111320 * Math.cos(lat * Math.PI / 180)),
      north: lat + radius / 111320,
      east: lon + radius / (111320 * Math.cos(lat * Math.PI / 180)),
    },
    timestamp: new Date().toISOString(),
  };
}

function haversine(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ============================================================================
// NASA Mars Rover API (Robotics Domain)
// https://api.nasa.gov — DEMO_KEY is built-in, free, public
// ============================================================================

const NASA_KEY = 'DEMO_KEY';

export interface RoverData {
  rover_name: string;
  rover_status: string;
  landing_date: string;
  max_sol: number;
  total_photos: number;
  cameras: string[];
  latest_photos: Array<{
    id: number;
    sol: number;
    camera: string;
    img_src: string;
    earth_date: string;
  }>;
  timestamp: string;
}

export async function fetchNASARover(rover: string = 'curiosity'): Promise<RoverData> {
  const manifestUrl = `https://api.nasa.gov/mars-photos/api/v1/manifests/${rover}?api_key=${NASA_KEY}`;
  const manifestRes = await fetch(manifestUrl, { next: { revalidate: 3600 } });
  if (!manifestRes.ok) throw new Error(`NASA API error: ${manifestRes.status}`);
  const manifestData = await manifestRes.json();
  const manifest = manifestData.photo_manifest;

  const latestSol = manifest.max_sol;
  const photosUrl = `https://api.nasa.gov/mars-photos/api/v1/rovers/${rover}/photos?sol=${latestSol}&page=1&api_key=${NASA_KEY}`;
  const photosRes = await fetch(photosUrl, { next: { revalidate: 3600 } });
  if (!photosRes.ok) throw new Error(`NASA Photos API error: ${photosRes.status}`);
  const photosData = await photosRes.json();

  return {
    rover_name: manifest.name,
    rover_status: manifest.status,
    landing_date: manifest.landing_date,
    max_sol: manifest.max_sol,
    total_photos: manifest.total_photos,
    cameras: (manifest.photos?.[manifest.photos.length - 1]?.cameras ?? []),
    latest_photos: (photosData.photos ?? []).slice(0, 10).map((p: any) => ({
      id: p.id,
      sol: p.sol,
      camera: p.camera?.name ?? 'Unknown',
      img_src: p.img_src,
      earth_date: p.earth_date,
    })),
    timestamp: new Date().toISOString(),
  };
}

// ============================================================================
// OpenSky Network API (Aviation / Drone Domain)
// https://opensky-network.org — free anonymous access, no API key
// ============================================================================

export interface AircraftState {
  icao24: string;
  callsign: string | null;
  origin_country: string;
  longitude: number | null;
  latitude: number | null;
  altitude_m: number | null;
  velocity_ms: number | null;
  heading: number | null;
  vertical_rate: number | null;
  on_ground: boolean;
}

export interface AviationData {
  aircraft_count: number;
  aircraft: AircraftState[];
  bounds: { lamin: number; lamax: number; lomin: number; lomax: number };
  timestamp: string;
  coordination_metrics: {
    avg_altitude_m: number;
    avg_velocity_ms: number;
    density_per_sq_km: number;
    airborne_count: number;
    grounded_count: number;
  };
}

export async function fetchAviationData(
  lamin: number = 40.5, lomin: number = -74.5, lamax: number = 41.0, lomax: number = -73.5
): Promise<AviationData> {
  const url = `https://opensky-network.org/api/states/all?lamin=${lamin}&lomin=${lomin}&lamax=${lamax}&lomax=${lomax}`;
  const res = await fetch(url, { next: { revalidate: 15 } });
  if (!res.ok) throw new Error(`OpenSky API error: ${res.status}`);
  const data = await res.json();

  const aircraft: AircraftState[] = (data.states ?? []).slice(0, 200).map((s: any[]) => ({
    icao24: s[0],
    callsign: s[1]?.trim() || null,
    origin_country: s[2],
    longitude: s[5],
    latitude: s[6],
    altitude_m: s[7],
    velocity_ms: s[9],
    heading: s[10],
    vertical_rate: s[11],
    on_ground: s[8],
  }));

  const airborne = aircraft.filter(a => !a.on_ground);
  const altitudes = airborne.map(a => a.altitude_m).filter(Boolean) as number[];
  const velocities = airborne.map(a => a.velocity_ms).filter(Boolean) as number[];
  const area = Math.abs((lamax - lamin) * (lomax - lomin)) * 111 * 111;

  return {
    aircraft_count: aircraft.length,
    aircraft,
    bounds: { lamin, lamax, lomin, lomax },
    timestamp: new Date().toISOString(),
    coordination_metrics: {
      avg_altitude_m: altitudes.length > 0 ? altitudes.reduce((a, b) => a + b, 0) / altitudes.length : 0,
      avg_velocity_ms: velocities.length > 0 ? velocities.reduce((a, b) => a + b, 0) / velocities.length : 0,
      density_per_sq_km: area > 0 ? aircraft.length / area : 0,
      airborne_count: airborne.length,
      grounded_count: aircraft.length - airborne.length,
    },
  };
}

// ============================================================================
// USGS Earthquake API (Research / Seismic Domain)
// https://earthquake.usgs.gov — completely free, no API key
// ============================================================================

export interface SeismicEvent {
  id: string;
  magnitude: number;
  place: string;
  time: string;
  latitude: number;
  longitude: number;
  depth_km: number;
  type: string;
  status: string;
  tsunami: boolean;
}

export interface SeismicData {
  event_count: number;
  events: SeismicEvent[];
  period: string;
  coordination_scenario: {
    total_agents_needed: number;
    urgency_score: number;
    affected_area_km2: number;
    max_magnitude: number;
    cooperation_gamma_recommended: number;
  };
  timestamp: string;
}

export async function fetchSeismicData(
  period: 'hour' | 'day' | 'week' = 'day',
  minMagnitude: number = 2.5
): Promise<SeismicData> {
  const periodMap = { hour: 'all_hour', day: 'all_day', week: 'all_week' };
  const url = `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/${periodMap[period]}.geojson`;
  const res = await fetch(url, { next: { revalidate: 300 } });
  if (!res.ok) throw new Error(`USGS API error: ${res.status}`);
  const data = await res.json();

  const events: SeismicEvent[] = (data.features ?? [])
    .filter((f: any) => (f.properties?.mag ?? 0) >= minMagnitude)
    .slice(0, 100)
    .map((f: any) => ({
      id: f.id,
      magnitude: f.properties.mag,
      place: f.properties.place,
      time: new Date(f.properties.time).toISOString(),
      latitude: f.geometry.coordinates[1],
      longitude: f.geometry.coordinates[0],
      depth_km: f.geometry.coordinates[2],
      type: f.properties.type,
      status: f.properties.status,
      tsunami: f.properties.tsunami === 1,
    }));

  const maxMag = events.length > 0 ? Math.max(...events.map(e => e.magnitude)) : 0;
  const urgency = Math.min(1, maxMag / 8);

  return {
    event_count: events.length,
    events,
    period,
    coordination_scenario: {
      total_agents_needed: Math.round(events.length * 50 * (1 + urgency)),
      urgency_score: urgency,
      affected_area_km2: events.length * 100 * (1 + maxMag),
      max_magnitude: maxMag,
      cooperation_gamma_recommended: 1.2 + urgency * 0.8,
    },
    timestamp: new Date().toISOString(),
  };
}

// ============================================================================
// World Bank API (Economic / Population Domain)
// https://api.worldbank.org — completely free, no API key
// ============================================================================

export interface EconomicData {
  country: string;
  country_code: string;
  population: number | null;
  gdp_usd: number | null;
  gdp_per_capita: number | null;
  urban_population_pct: number | null;
  year: number;
  agent_modeling: {
    suggested_N: number;
    cooperation_proxy: number;
    network_density_proxy: number;
  };
  timestamp: string;
}

export async function fetchEconomicData(
  countryCode: string = 'USA',
  year: number = 2023
): Promise<EconomicData> {
  const indicators = [
    'SP.POP.TOTL',        // Population
    'NY.GDP.MKTP.CD',     // GDP (current USD)
    'NY.GDP.PCAP.CD',     // GDP per capita
    'SP.URB.TOTL.IN.ZS',  // Urban population %
  ];

  const results: Record<string, number | null> = {};

  for (const indicator of indicators) {
    const url = `https://api.worldbank.org/v2/country/${countryCode}/indicator/${indicator}?date=${year - 2}:${year}&format=json&per_page=5`;
    try {
      const res = await fetch(url, { next: { revalidate: 86400 } });
      if (res.ok) {
        const data = await res.json();
        const entries = data[1] ?? [];
        const latest = entries.find((e: any) => e.value !== null);
        results[indicator] = latest?.value ?? null;
      }
    } catch {
      results[indicator] = null;
    }
  }

  const pop = results['SP.POP.TOTL'];
  const gdp = results['NY.GDP.MKTP.CD'];
  const gdpPc = results['NY.GDP.PCAP.CD'];
  const urban = results['SP.URB.TOTL.IN.ZS'];

  const suggestedN = pop ? Math.min(100000, Math.max(100, Math.round(pop / 10000))) : 10000;
  const cooperationProxy = gdpPc ? Math.min(1, gdpPc / 80000) : 0.5;
  const densityProxy = urban ? urban / 100 : 0.5;

  return {
    country: countryCode,
    country_code: countryCode,
    population: pop,
    gdp_usd: gdp,
    gdp_per_capita: gdpPc,
    urban_population_pct: urban,
    year,
    agent_modeling: {
      suggested_N: suggestedN,
      cooperation_proxy: cooperationProxy,
      network_density_proxy: densityProxy,
    },
    timestamp: new Date().toISOString(),
  };
}
