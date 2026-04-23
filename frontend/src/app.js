import "leaflet/dist/leaflet.css";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  Polygon,
  CircleMarker,
  Popup,
  useMapEvents,
} from "react-leaflet";
import { useEffect, useMemo, useState } from "react";

const AUSTIN_CENTER = [30.2672, -97.7431];

// giant outer rectangle covering the whole map world
const OUTER_RING = [
  [90, -180],
  [90, 180],
  [-90, 180],
  [-90, -180],
];

// convert GeoJSON geometry into Leaflet-style hole rings
function getCountyOuterRings(geojson) {
  if (!geojson || !geojson.features || geojson.features.length === 0) return [];

  const geometry = geojson.features[0].geometry;

  if (geometry.type === "Polygon") {
    // only outer ring
    return [geometry.coordinates[0].map(([lng, lat]) => [lat, lng])];
  }

  if (geometry.type === "MultiPolygon") {
    // each polygon -> take its outer ring
    return geometry.coordinates.map((polygon) =>
      polygon[0].map(([lng, lat]) => [lat, lng])
    );
  }

  return [];
}

function ClickPoint({ selectedPoint, setSelectedPoint }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      setSelectedPoint({
        lat: Number(lat.toFixed(6)),
        lon: Number(lng.toFixed(6)),
      });
    },
  });

  if (!selectedPoint) return null;

  return (
    <CircleMarker
      center={[selectedPoint.lat, selectedPoint.lon]}
      radius={6}
      pathOptions={{
        color: "black",
        weight: 1,
        fillColor: "black",
        fillOpacity: 1,
      }}
    >
      <Popup>
        Candidate station
        <br />
        Lat: {selectedPoint.lat}
        <br />
        Lon: {selectedPoint.lon}
      </Popup>
    </CircleMarker>
  );
}

export default function App() {
  const [countyGeoJson, setCountyGeoJson] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);

  useEffect(() => {
    fetch("/travis_county.geojson")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        console.log("Loaded Travis County GeoJSON:", data);
        setCountyGeoJson(data);
      })
      .catch((err) => {
        console.error("Error loading Travis County GeoJSON:", err);
      });
  }, []);

  const countyHoles = useMemo(() => {
    return getCountyOuterRings(countyGeoJson);
  }, [countyGeoJson]);

  return (
    <div style={{ height: "100vh", width: "100%" }}>
      <MapContainer
        center={AUSTIN_CENTER}
        zoom={10}
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* White transparent mask outside Travis County */}
        {countyHoles.length > 0 && (
          <Polygon
            positions={[OUTER_RING, ...countyHoles]}
            pathOptions={{
              stroke: false,
              fillColor: "white",
              fillOpacity: 0.3,
              fillRule: "evenodd",
              interactive: false,
            }}
          />
        )}

        {/* Travis County border */}
        {countyGeoJson && (
          <GeoJSON
            data={countyGeoJson}
            style={() => ({
              color: "black",
              weight: 2,
              fill: false,
            })}
          />
        )}

        {/* Clickable point */}
        <ClickPoint
          selectedPoint={selectedPoint}
          setSelectedPoint={setSelectedPoint}
        />
      </MapContainer>
    </div>
  );
}