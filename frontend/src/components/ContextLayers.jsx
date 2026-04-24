import React, { useEffect, useMemo, useState } from "react";
import { Marker, Tooltip } from "react-leaflet";
import L from "leaflet";
import Papa from "papaparse";
import "./ContextLayers.css";

const LAYER_CONFIGS = [
  {
    id: "transit",
    label: "Transit Stop",
    url: "/map_layers/transit_stops.csv",
    color: "#7c3aed",
    size: 9,
  },
  {
    id: "amenities",
    label: "Amenity",
    url: "/map_layers/amenities.csv",
    color: "#16a34a",
    size: 9,
  },
  {
    id: "jobs",
    label: "Jobs",
    url: "/map_layers/jobs.csv",
    color: "#f97316",
    size: 7,
  },
  {
    id: "bikeshare",
    label: "Existing Bikeshare Station",
    url: "/map_layers/existing_stations.csv",
    color: "#007bba",
    size: 12,
  },
];

function getDistanceMeters(lat1, lon1, lat2, lon2) {
  const earthRadiusMeters = 6371000;
  const toRadians = (degrees) => (degrees * Math.PI) / 180;

  const dLat = toRadians(lat2 - lat1);
  const dLon = toRadians(lon2 - lon1);

  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRadians(lat1)) *
      Math.cos(toRadians(lat2)) *
      Math.sin(dLon / 2) ** 2;

  return earthRadiusMeters * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function getNumber(row, possibleNames) {
  for (const name of possibleNames) {
    const value = row[name];

    if (value !== undefined && value !== null && value !== "") {
      const numericValue = Number(value);

      if (Number.isFinite(numericValue)) {
        return numericValue;
      }
    }
  }

  return null;
}

function getString(row, possibleNames, fallback = "") {
  for (const name of possibleNames) {
    const value = row[name];

    if (value !== undefined && value !== null && String(value).trim() !== "") {
      return String(value).trim();
    }
  }

  return fallback;
}

function formatName(name) {
  return String(name || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function normalizeRow(row, layer) {
  const lat = getNumber(row, ["lat", "latitude", "Latitude", "LAT"]);
  const lon = getNumber(row, ["lon", "lng", "longitude", "Longitude", "LON"]);

  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return null;
  }

  const rawName = getString(
    row,
    ["name", "Name", "station", "Station", "stop_name", "Stop Name"],
    layer.label
  );

  return {
    id: `${layer.id}-${rawName}-${lat}-${lon}`,
    layerId: layer.id,
    layerLabel: layer.label,
    name: formatName(rawName),
    lat,
    lon,
    type: getString(row, ["type", "Type", "category", "Category"], layer.label),
    docks: getNumber(row, ["docks", "Docks"]),
    trips: getNumber(row, ["trips", "Trips"]),
    tripsPerDock: getNumber(row, [
      "trips_per_dock",
      "Trips Per Dock",
      "tripsPerDock",
    ]),
    bikeableInfrastructure: getNumber(row, [
      "bikeable_infrastructure",
      "Bikeable Infrastructure",
    ]),
    color: layer.color,
    size: layer.size,
  };
}

function createContextPointIcon(point, delayMs) {
  return L.divIcon({
    className: "context-point-icon",
    html: `
      <span
        class="context-point-dot context-point-${point.layerId}"
        style="
          --context-color: ${point.color};
          --context-size: ${point.size}px;
          --context-delay: ${delayMs}ms;
        "
      ></span>
    `,
    iconSize: [point.size + 16, point.size + 16],
    iconAnchor: [(point.size + 16) / 2, (point.size + 16) / 2],
  });
}

function TooltipContent({ point }) {
  return (
    <div>
      <strong>{point.name}</strong>
      <br />
      {point.layerLabel}

      {point.type && (
        <>
          <br />
          Type: {formatName(point.type)}
        </>
      )}

      {point.docks !== null && (
        <>
          <br />
          Docks: {point.docks}
        </>
      )}

      {point.tripsPerDock !== null && (
        <>
          <br />
          Trips per dock: {Math.round(point.tripsPerDock).toLocaleString()}
        </>
      )}

      {point.trips !== null && (
        <>
          <br />
          Total trips: {Math.round(point.trips).toLocaleString()}
        </>
      )}

      <br />
      {Math.round(point.distanceMeters).toLocaleString()}m away
    </div>
  );
}

export default function ContextLayers({
  selectedPoint,
  showLayers,
  radiusMeters = 2000,
}) {
  const [pointsByLayer, setPointsByLayer] = useState({});

  useEffect(() => {
    async function loadLayers() {
      const loadedLayers = {};

      for (const layer of LAYER_CONFIGS) {
        try {
          const response = await fetch(layer.url);

          if (!response.ok) {
            console.warn(`Could not load ${layer.url}`);
            loadedLayers[layer.id] = [];
            continue;
          }

          const csvText = await response.text();

          const parsed = Papa.parse(csvText, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: false,
          });

          loadedLayers[layer.id] = parsed.data
            .map((row) => normalizeRow(row, layer))
            .filter(Boolean);
        } catch (error) {
          console.error(`Error loading layer ${layer.id}:`, error);
          loadedLayers[layer.id] = [];
        }
      }

      setPointsByLayer(loadedLayers);
    }

    loadLayers();
  }, []);

  const nearbyPoints = useMemo(() => {
    if (!selectedPoint || !showLayers) return [];

    return Object.values(pointsByLayer)
      .flat()
      .map((point) => {
        const distanceMeters = getDistanceMeters(
          selectedPoint.lat,
          selectedPoint.lon,
          point.lat,
          point.lon
        );

        return {
          ...point,
          distanceMeters,
        };
      })
      .filter((point) => point.distanceMeters <= radiusMeters)
      .sort((a, b) => a.distanceMeters - b.distanceMeters)
      .map((point, index) => ({
        ...point,
        animationDelayMs: Math.min(index * 28, 1200),
      }));
  }, [pointsByLayer, selectedPoint, showLayers, radiusMeters]);

  if (!selectedPoint || !showLayers) return null;

  return (
    <>
      {nearbyPoints.map((point) => (
        <Marker
          key={`${point.id}-${selectedPoint.id}`}
          position={[point.lat, point.lon]}
          icon={createContextPointIcon(point, point.animationDelayMs)}
          interactive
        >
          <Tooltip direction="top" offset={[0, -8]}>
            <TooltipContent point={point} />
          </Tooltip>
        </Marker>
      ))}
    </>
  );
}