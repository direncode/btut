#!/usr/bin/env python3
"""
OSM Network Builder for BTUT-SUMO

Downloads and converts OpenStreetMap data to SUMO networks.
Enables testing BTUT on real city layouts.

Usage:
    # Build Chapel Hill downtown network
    python osm_network_builder.py --city "Chapel Hill, NC" --radius 1000

    # Build from bounding box
    python osm_network_builder.py --bbox "-79.06,35.90,-79.04,35.92" --name downtown

    # Build from existing OSM file
    python osm_network_builder.py --osm-file city.osm --name my_city
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class OSMNetworkBuilder:
    """
    Builds SUMO networks from OpenStreetMap data.

    Workflow:
    1. Download OSM data (via Overpass API or local file)
    2. Convert to SUMO network (netconvert)
    3. Generate random traffic (randomTrips.py)
    4. Create SUMO config file
    """

    def __init__(self, output_dir: str = "networks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Find SUMO_HOME
        self.sumo_home = os.environ.get("SUMO_HOME", "")
        if not self.sumo_home:
            # Try common locations
            common_paths = [
                "/usr/share/sumo",
                "/usr/local/opt/sumo/share/sumo",
                "C:\\Program Files (x86)\\Eclipse\\Sumo",
                "C:\\Program Files\\Eclipse\\Sumo",
            ]
            for path in common_paths:
                if os.path.exists(path):
                    self.sumo_home = path
                    break

        if self.sumo_home:
            logger.info(f"SUMO_HOME: {self.sumo_home}")
        else:
            logger.warning("SUMO_HOME not found. Some features may not work.")

    def download_osm_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        output_file: str
    ) -> bool:
        """
        Download OSM data for a bounding box.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            output_file: Output .osm file path
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        # Overpass API query
        query = f"""
        [out:xml][timeout:300];
        (
          way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
          node(w);
        );
        out body;
        """

        logger.info(f"Downloading OSM data for bbox: {bbox}")

        try:
            import requests

            response = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                timeout=300
            )

            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded to {output_file}")
                return True
            else:
                logger.error(f"Download failed: {response.status_code}")
                return False

        except ImportError:
            logger.error("requests not installed. Install with: pip install requests")
            return False
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False

    def download_osm_city(
        self,
        city_name: str,
        radius_m: int = 1000,
        output_file: str = None
    ) -> Optional[str]:
        """
        Download OSM data for a city by name.

        Args:
            city_name: City name (e.g., "Chapel Hill, NC")
            radius_m: Radius in meters around city center
            output_file: Output file path (auto-generated if None)

        Returns:
            Path to downloaded file, or None on failure
        """
        try:
            import requests

            # Geocode city name
            logger.info(f"Geocoding: {city_name}")
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": city_name,
                    "format": "json",
                    "limit": 1
                },
                headers={"User-Agent": "BTUT-SUMO/1.0"}
            )

            if response.status_code != 200 or not response.json():
                logger.error(f"Geocoding failed for: {city_name}")
                return None

            result = response.json()[0]
            lat = float(result["lat"])
            lon = float(result["lon"])

            logger.info(f"Found: {result['display_name']}")
            logger.info(f"Center: ({lat}, {lon})")

            # Calculate bbox from radius
            # Approximate: 1 degree lat ~ 111km, 1 degree lon ~ 111km * cos(lat)
            import math
            lat_offset = radius_m / 111000
            lon_offset = radius_m / (111000 * math.cos(math.radians(lat)))

            bbox = (
                lon - lon_offset,
                lat - lat_offset,
                lon + lon_offset,
                lat + lat_offset
            )

            if output_file is None:
                safe_name = city_name.replace(" ", "_").replace(",", "").lower()
                output_file = str(self.output_dir / f"{safe_name}.osm")

            if self.download_osm_bbox(bbox, output_file):
                return output_file
            return None

        except ImportError:
            logger.error("requests not installed")
            return None
        except Exception as e:
            logger.error(f"Error: {e}")
            return None

    def convert_to_sumo(
        self,
        osm_file: str,
        network_name: str,
        keep_all_lanes: bool = False
    ) -> Optional[str]:
        """
        Convert OSM file to SUMO network.

        Args:
            osm_file: Input .osm file
            network_name: Name for output files
            keep_all_lanes: If True, keep all lane types (slower)

        Returns:
            Path to generated .net.xml file
        """
        output_net = self.output_dir / f"{network_name}.net.xml"

        cmd = [
            "netconvert",
            "--osm-files", osm_file,
            "--output-file", str(output_net),
            "--geometry.remove", "true",
            "--roundabouts.guess", "true",
            "--ramps.guess", "true",
            "--junctions.join", "true",
            "--tls.guess-signals", "true",
            "--tls.discard-simple", "true",
            "--tls.join", "true",
        ]

        if not keep_all_lanes:
            cmd.extend([
                "--remove-edges.isolated", "true",
                "--keep-edges.by-vclass", "passenger",
            ])

        logger.info(f"Converting {osm_file} to SUMO network...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                logger.info(f"Network created: {output_net}")
                return str(output_net)
            else:
                logger.error(f"netconvert failed: {result.stderr}")
                return None

        except FileNotFoundError:
            logger.error("netconvert not found. Is SUMO installed?")
            return None
        except subprocess.TimeoutExpired:
            logger.error("netconvert timed out")
            return None

    def generate_routes(
        self,
        network_file: str,
        network_name: str,
        num_trips: int = 1000,
        period: float = 1.0,
        duration: int = 3600
    ) -> Optional[str]:
        """
        Generate random routes for the network.

        Args:
            network_file: Input .net.xml file
            network_name: Name for output files
            num_trips: Approximate number of trips to generate
            period: Average time between vehicle departures
            duration: Simulation duration in seconds

        Returns:
            Path to generated routes file
        """
        output_routes = self.output_dir / f"{network_name}.rou.xml"

        # Find randomTrips.py
        random_trips = None
        if self.sumo_home:
            random_trips = Path(self.sumo_home) / "tools" / "randomTrips.py"
            if not random_trips.exists():
                random_trips = None

        if not random_trips:
            logger.error("randomTrips.py not found")
            return None

        cmd = [
            sys.executable, str(random_trips),
            "-n", network_file,
            "-o", str(output_routes),
            "-e", str(duration),
            "-p", str(period),
            "--validate",
            "--route-file", str(output_routes),
        ]

        logger.info(f"Generating routes for {network_name}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0 or output_routes.exists():
                logger.info(f"Routes created: {output_routes}")
                return str(output_routes)
            else:
                logger.error(f"Route generation failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Route generation timed out")
            return None

    def create_config(
        self,
        network_file: str,
        routes_file: str,
        network_name: str,
        duration: int = 3600
    ) -> str:
        """
        Create SUMO configuration file.

        Returns:
            Path to generated .sumocfg file
        """
        config_file = self.output_dir / f"{network_name}.sumocfg"

        config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{Path(network_file).name}"/>
        <route-files value="{Path(routes_file).name}"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="{duration}"/>
        <step-length value="1"/>
    </time>

    <processing>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
        <time-to-teleport value="-1"/>
    </processing>

    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
        <duration-log.statistics value="true"/>
    </report>

    <gui_only>
        <start value="true"/>
        <quit-on-end value="true"/>
    </gui_only>

</configuration>
"""

        with open(config_file, 'w') as f:
            f.write(config_content)

        logger.info(f"Config created: {config_file}")
        return str(config_file)

    def build_network(
        self,
        city_name: str = None,
        bbox: Tuple[float, float, float, float] = None,
        osm_file: str = None,
        network_name: str = None,
        radius_m: int = 1000,
        num_trips: int = 1000,
        duration: int = 3600
    ) -> Optional[dict]:
        """
        Complete workflow: download/load OSM -> convert -> generate routes -> config.

        Args:
            city_name: City to download (e.g., "Chapel Hill, NC")
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            osm_file: Existing OSM file
            network_name: Name for output files
            radius_m: Radius for city download
            num_trips: Number of trips to generate
            duration: Simulation duration

        Returns:
            Dict with paths to all generated files
        """
        # Determine OSM source
        if osm_file:
            osm_path = osm_file
            if not network_name:
                network_name = Path(osm_file).stem
        elif city_name:
            if not network_name:
                network_name = city_name.replace(" ", "_").replace(",", "").lower()
            osm_path = self.download_osm_city(city_name, radius_m)
            if not osm_path:
                return None
        elif bbox:
            if not network_name:
                network_name = "custom_area"
            osm_path = str(self.output_dir / f"{network_name}.osm")
            if not self.download_osm_bbox(bbox, osm_path):
                return None
        else:
            logger.error("Must specify city_name, bbox, or osm_file")
            return None

        # Convert to SUMO network
        net_file = self.convert_to_sumo(osm_path, network_name)
        if not net_file:
            return None

        # Generate routes
        routes_file = self.generate_routes(
            net_file, network_name,
            num_trips=num_trips,
            duration=duration
        )
        if not routes_file:
            return None

        # Create config
        config_file = self.create_config(
            net_file, routes_file, network_name, duration
        )

        return {
            "osm_file": osm_path,
            "network_file": net_file,
            "routes_file": routes_file,
            "config_file": config_file,
            "network_name": network_name
        }


# Pre-defined city configurations for quick testing
PRESET_CITIES = {
    "chapel_hill": {
        "city_name": "Chapel Hill, NC",
        "radius_m": 1500,
        "num_trips": 2000
    },
    "downtown_durham": {
        "city_name": "Durham, NC",
        "radius_m": 1000,
        "num_trips": 1500
    },
    "san_francisco": {
        "bbox": (-122.42, 37.77, -122.40, 37.79),
        "num_trips": 3000
    },
    "manhattan_midtown": {
        "bbox": (-73.99, 40.75, -73.97, 40.77),
        "num_trips": 5000
    },
    "london_city": {
        "bbox": (-0.10, 51.51, -0.07, 51.52),
        "num_trips": 3000
    }
}


def main():
    parser = argparse.ArgumentParser(
        description='Build SUMO networks from OpenStreetMap'
    )

    # Source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        '--city', type=str,
        help='City name to download (e.g., "Chapel Hill, NC")'
    )
    source_group.add_argument(
        '--bbox', type=str,
        help='Bounding box: "min_lon,min_lat,max_lon,max_lat"'
    )
    source_group.add_argument(
        '--osm-file', type=str,
        help='Existing OSM file'
    )
    source_group.add_argument(
        '--preset', type=str, choices=list(PRESET_CITIES.keys()),
        help='Use preset city configuration'
    )

    # Options
    parser.add_argument(
        '--name', type=str,
        help='Network name (auto-generated if not specified)'
    )
    parser.add_argument(
        '--radius', type=int, default=1000,
        help='Radius in meters for city download (default: 1000)'
    )
    parser.add_argument(
        '--trips', type=int, default=1000,
        help='Number of trips to generate (default: 1000)'
    )
    parser.add_argument(
        '--duration', type=int, default=3600,
        help='Simulation duration in seconds (default: 3600)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='networks',
        help='Output directory (default: networks)'
    )
    parser.add_argument(
        '--list-presets', action='store_true',
        help='List available preset cities'
    )

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print("\nAvailable preset cities:")
        print("-" * 50)
        for name, config in PRESET_CITIES.items():
            if 'city_name' in config:
                print(f"  {name}: {config['city_name']}")
            else:
                print(f"  {name}: bbox {config['bbox']}")
        print("\nUsage: python osm_network_builder.py --preset chapel_hill")
        return

    # Build network
    builder = OSMNetworkBuilder(output_dir=args.output_dir)

    if args.preset:
        config = PRESET_CITIES[args.preset]
        result = builder.build_network(
            city_name=config.get('city_name'),
            bbox=config.get('bbox'),
            network_name=args.name or args.preset,
            radius_m=config.get('radius_m', args.radius),
            num_trips=config.get('num_trips', args.trips),
            duration=args.duration
        )
    elif args.city:
        result = builder.build_network(
            city_name=args.city,
            network_name=args.name,
            radius_m=args.radius,
            num_trips=args.trips,
            duration=args.duration
        )
    elif args.bbox:
        bbox = tuple(map(float, args.bbox.split(',')))
        result = builder.build_network(
            bbox=bbox,
            network_name=args.name or "custom",
            num_trips=args.trips,
            duration=args.duration
        )
    elif args.osm_file:
        result = builder.build_network(
            osm_file=args.osm_file,
            network_name=args.name,
            num_trips=args.trips,
            duration=args.duration
        )
    else:
        parser.print_help()
        print("\nExample: python osm_network_builder.py --city 'Chapel Hill, NC' --radius 1500")
        return

    if result:
        print("\n" + "=" * 60)
        print("NETWORK BUILD COMPLETE")
        print("=" * 60)
        print(f"Network: {result['network_name']}")
        print(f"Config:  {result['config_file']}")
        print(f"\nTo run with BTUT:")
        print(f"  python btut_sumo_bridge.py --config {result['config_file']} --experiment comparison")
        print("=" * 60)


if __name__ == '__main__':
    main()
